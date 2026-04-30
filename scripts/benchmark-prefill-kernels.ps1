param(
    [string]$Executable = "build/qwen35x.exe",
    [string]$HFModelDir = "models/qwen3.5-0.8b",
    [string]$PromptFile = "benchmarks/inputs/wiki_artificial_intelligence_64k_prompt.txt",
    [string]$CsvOut = "benchmarks/qwen35x-prefill-kernel-head-to-head.csv",
    [string[]]$PrefillKernels = @("traditional", "flashqla"),
    [int[]]$TargetPromptTokens = @(256, 1024, 4096, 16384, 65536),
    [int]$Runs = 3,
    [int]$WarmupRuns = 1,
    [int]$MaxNewTokens = 128,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath {
    param(
        [string]$Path,
        [string]$RepoRoot
    )
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return Join-Path $RepoRoot $Path
}

function ConvertTo-InvariantString {
    param([object]$Value)
    if ($null -eq $Value) {
        return ""
    }
    if ($Value -is [double] -or $Value -is [float] -or $Value -is [decimal]) {
        return ([double]$Value).ToString("G17", [System.Globalization.CultureInfo]::InvariantCulture)
    }
    return [string]$Value
}

function Parse-MetricsLine {
    param([string]$Line)
    $map = @{}
    foreach ($match in [regex]::Matches($Line, '([A-Za-z_]+)=([^ ]+)')) {
        $map[$match.Groups[1].Value] = $match.Groups[2].Value
    }
    return $map
}

function New-TruncatedPrompt {
    param(
        [string]$SourceText,
        [int]$TargetTokens,
        [int]$ReferenceTokens,
        [string]$OutPath
    )
    $ratio = [Math]::Min(1.0, [double]$TargetTokens / [double]$ReferenceTokens)
    $charCount = [Math]::Max(16, [Math]::Min($SourceText.Length, [int][Math]::Ceiling($SourceText.Length * $ratio)))
    Set-Content -LiteralPath $OutPath -Value $SourceText.Substring(0, $charCount) -NoNewline -Encoding UTF8
}

function Invoke-OneRun {
    param(
        [string]$Exe,
        [string]$ModelDir,
        [string]$PromptPath,
        [string]$Kernel,
        [int]$MaxContext,
        [int]$MaxNewTokens
    )
    $args = @(
        "--infer-gpu",
        "--hf-model-dir", $ModelDir,
        "--prompt-file", $PromptPath,
        "--max-new-tokens", "$MaxNewTokens",
        "--max-context", "$MaxContext",
        "--temperature", "0",
        "--qwen35x-prefill-mode", "batched",
        "--qwen35x-prefill-kernel", $Kernel,
        "--metrics-only"
    )
    $output = & $Exe @args 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "qwen35x failed for kernel=$Kernel max_context=$MaxContext`n$output"
    }
    $line = ($output | Where-Object { $_ -match '^backend=' } | Select-Object -Last 1)
    if ([string]::IsNullOrWhiteSpace($line)) {
        throw "qwen35x did not emit a metrics line for kernel=$Kernel.`n$output"
    }
    return Parse-MetricsLine -Line $line
}

$scriptDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $scriptDir
$resolvedExe = Resolve-RepoPath -Path $Executable -RepoRoot $repoRoot
$resolvedModelDir = Resolve-RepoPath -Path $HFModelDir -RepoRoot $repoRoot
$resolvedPromptFile = Resolve-RepoPath -Path $PromptFile -RepoRoot $repoRoot
$resolvedCsvOut = Resolve-RepoPath -Path $CsvOut -RepoRoot $repoRoot
$csvDir = Split-Path -Parent $resolvedCsvOut
if (-not (Test-Path $csvDir)) {
    New-Item -ItemType Directory -Path $csvDir | Out-Null
}

if (-not $SkipBuild) {
    & (Join-Path $scriptDir "build.ps1") -UseNinja -EnableCuda -Configuration Release -Target qwen35x
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed."
    }
}

if (-not (Test-Path $resolvedExe)) {
    throw "Executable not found: $resolvedExe"
}
if (-not (Test-Path $resolvedPromptFile)) {
    throw "Prompt file not found: $resolvedPromptFile"
}

$sourceText = Get-Content -Raw -LiteralPath $resolvedPromptFile
$referenceTokens = 65405
$tempDir = Join-Path $repoRoot "build\prefill-kernel-prompts"
if (-not (Test-Path $tempDir)) {
    New-Item -ItemType Directory -Path $tempDir | Out-Null
}

$rows = New-Object System.Collections.Generic.List[object]
Write-Host "Prefill kernel head-to-head benchmark" -ForegroundColor Green
Write-Host "Executable: $resolvedExe"
Write-Host "CSV: $resolvedCsvOut"

foreach ($targetTokens in $TargetPromptTokens) {
    $promptPath = Join-Path $tempDir ("prompt-target-{0}.txt" -f $targetTokens)
    New-TruncatedPrompt -SourceText $sourceText -TargetTokens $targetTokens -ReferenceTokens $referenceTokens -OutPath $promptPath
    $maxContext = $targetTokens + $MaxNewTokens + 256

    foreach ($kernel in $PrefillKernels) {
        for ($warmup = 1; $warmup -le $WarmupRuns; ++$warmup) {
            [void](Invoke-OneRun -Exe $resolvedExe -ModelDir $resolvedModelDir -PromptPath $promptPath -Kernel $kernel -MaxContext $maxContext -MaxNewTokens $MaxNewTokens)
            Write-Host ("Warmup complete: target={0} kernel={1} {2}/{3}" -f $targetTokens, $kernel, $warmup, $WarmupRuns)
        }

        for ($run = 1; $run -le $Runs; ++$run) {
            $metrics = Invoke-OneRun -Exe $resolvedExe -ModelDir $resolvedModelDir -PromptPath $promptPath -Kernel $kernel -MaxContext $maxContext -MaxNewTokens $MaxNewTokens
            $row = [pscustomobject]@{
                target_prompt_tokens = $targetTokens
                actual_prompt_tokens = $metrics["prompt_tokens"]
                kernel = $kernel
                run = $run
                generated_tokens = $metrics["generated_tokens"]
                prefill_time_ms = $metrics["prefill_time_ms"]
                prefill_tps = $metrics["prefill_tps"]
                decode_time_ms = $metrics["decode_time_ms"]
                decode_tps = $metrics["decode_tps"]
            }
            $rows.Add($row)
            Write-Host ("Recorded: target={0} actual={1} kernel={2} run={3}/{4} prefill_tps={5} decode_tps={6}" -f `
                $targetTokens, $row.actual_prompt_tokens, $kernel, $run, $Runs, $row.prefill_tps, $row.decode_tps) -ForegroundColor Yellow
        }
    }
}

$rows | Export-Csv -LiteralPath $resolvedCsvOut -NoTypeInformation
Write-Host "Benchmark complete." -ForegroundColor Green
Write-Host "CSV written to: $resolvedCsvOut"
