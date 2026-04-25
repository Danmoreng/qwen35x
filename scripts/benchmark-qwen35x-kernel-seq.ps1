[CmdletBinding()]
param(
    [string]$Executable = "build/qwen35x_kernelbench.exe",
    [string]$HFModelDir = "models/qwen3.5-0.8b",
    [string]$CsvOut = "benchmarks/qwen35x-kernel-seq.csv",
    [string]$RunLabel = "",
    [string]$PromptText = "Hello",
    [string]$LongPromptText = "Explain in great detail the history of artificial intelligence, machine learning, deep learning, and neural networks. ",
    [int]$Runs = 3,
    [int]$WarmupRuns = 1,
    [int]$MaxNewTokens = 128,
    [int]$MaxContext = 256,
    [int]$DecodeBlocks = 52
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$RepoRoot
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $RepoRoot $Path)
}

function To-InvariantString {
    param([Parameter(Mandatory = $true)]$Value)
    return [System.Convert]::ToString($Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Invoke-Qwen35xKernelRun {
    param(
        [Parameter(Mandatory = $true)][string]$ExePath,
        [Parameter(Mandatory = $true)][string]$ModelDir,
        [Parameter(Mandatory = $true)][string]$PromptText,
        [Parameter(Mandatory = $true)][string]$LongPromptText,
        [Parameter(Mandatory = $true)][int]$MaxNewTokens,
        [Parameter(Mandatory = $true)][int]$MaxContext,
        [Parameter(Mandatory = $true)][int]$WarmupRuns,
        [Parameter(Mandatory = $true)][int]$Runs,
        [Parameter(Mandatory = $true)][int]$DecodeBlocks,
        [Parameter(Mandatory = $true)][string]$ProfileJsonPath
    )

    $args = @(
        "--hf-model-dir", $ModelDir,
        "--prompt-text", $PromptText,
        "--long-prompt-text", $LongPromptText,
        "--max-new-tokens", "$MaxNewTokens",
        "--max-context", "$MaxContext",
        "--warmup-runs", "$WarmupRuns",
        "--runs", "$Runs",
        "--profile-json", $ProfileJsonPath
    )
    if ($DecodeBlocks -gt 0) {
        $args += @("--decode-blocks", "$DecodeBlocks")
    }

    Write-Host "Running qwen35x kernel benchmark..." -ForegroundColor Cyan
    $runOutput = & $ExePath @args 2>&1
    foreach ($line in $runOutput) {
        Write-Host $line
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Qwen35x kernel benchmark run failed (exit_code=$LASTEXITCODE)."
    }

    if (-not (Test-Path $ProfileJsonPath)) {
        throw "Missing profile JSON output: $ProfileJsonPath"
    }

    return Get-Content -Raw -LiteralPath $ProfileJsonPath | ConvertFrom-Json
}

$scriptDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $scriptDir

$resolvedExe = Resolve-RepoPath -Path $Executable -RepoRoot $repoRoot
$resolvedModelDir = Resolve-RepoPath -Path $HFModelDir -RepoRoot $repoRoot
$resolvedCsvOut = Resolve-RepoPath -Path $CsvOut -RepoRoot $repoRoot
$profileTmpDir = Join-Path $repoRoot "build\bench-profiles"

if (-not (Test-Path $resolvedExe)) {
    throw "Executable not found: $resolvedExe"
}
if (-not (Test-Path $resolvedModelDir)) {
    throw "Model directory not found: $resolvedModelDir"
}
if ($Runs -lt 1) {
    throw "Runs must be >= 1."
}
if ($WarmupRuns -lt 0) {
    throw "WarmupRuns must be >= 0."
}
if ($MaxNewTokens -lt 1) {
    throw "MaxNewTokens must be >= 1."
}
if ($MaxContext -lt 1) {
    throw "MaxContext must be >= 1."
}
if ($DecodeBlocks -lt 0) {
    throw "DecodeBlocks must be >= 0."
}

New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedCsvOut) -Force | Out-Null
New-Item -ItemType Directory -Path $profileTmpDir -Force | Out-Null

Write-Host ("Sequential qwen35x kernel benchmark start: warmup={0}, runs={1}" -f $WarmupRuns, $Runs) -ForegroundColor Green
Write-Host ("CSV output: {0}" -f $resolvedCsvOut) -ForegroundColor Green

for ($runIndex = 1; $runIndex -le $Runs; ++$runIndex) {
    $profilePath = Join-Path $profileTmpDir ("qwen35x_kernel_run_{0}_{1}.json" -f $runIndex, [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())
    try {
        $profile = Invoke-Qwen35xKernelRun `
            -ExePath $resolvedExe `
            -ModelDir $resolvedModelDir `
            -PromptText $PromptText `
            -LongPromptText $LongPromptText `
            -MaxNewTokens $MaxNewTokens `
            -MaxContext $MaxContext `
            -WarmupRuns $WarmupRuns `
            -Runs 1 `
            -DecodeBlocks $DecodeBlocks `
            -ProfileJsonPath $profilePath

        $row = [PSCustomObject]@{
            timestamp_utc         = [DateTime]::UtcNow.ToString("o")
            run_label             = $RunLabel
            mode                  = "qwen35x-kernel-cuda"
            run_index             = $runIndex
            prompt_tokens         = [int]$profile.prompt_tokens
            prefill_prompt_tokens = [int]$profile.prefill_prompt_tokens
            generated_tokens      = [int]$profile.generated_tokens
            load_time_ms          = To-InvariantString $profile.load_time_ms
            pp_time_ms            = To-InvariantString $profile.pp_time_ms
            pp_tokens_per_second  = To-InvariantString $profile.pp_tokens_per_second
            decode_time_ms        = To-InvariantString $profile.decode_time_ms
            tokens_per_second     = To-InvariantString $profile.tokens_per_second
            decode_blocks         = $DecodeBlocks
        }

        if (Test-Path $resolvedCsvOut) {
            $row | Export-Csv -LiteralPath $resolvedCsvOut -NoTypeInformation -Append
        } else {
            $row | Export-Csv -LiteralPath $resolvedCsvOut -NoTypeInformation
        }

        Write-Host ("Recorded: run={0}/{1} tg_tps={2}" -f $runIndex, $Runs, $row.tokens_per_second) -ForegroundColor Yellow
    } finally {
        if (Test-Path $profilePath) {
            Remove-Item -LiteralPath $profilePath -Force
        }
    }
}

Write-Host "Qwen35x kernel benchmark complete." -ForegroundColor Green
Write-Host ("CSV written to: {0}" -f $resolvedCsvOut) -ForegroundColor Green
