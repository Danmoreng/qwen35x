[CmdletBinding()]
param(
    [string]$Executable = "build/qwen35x.exe",
    [string]$HFModelDir = "models/qwen3.5-0.8b",
    [string]$CsvOut = "benchmarks/qwen35x-inference-seq.csv",
    [string]$RunLabel = "",
    [ValidateSet("gpu-bf16", "gpu-f32", "cpu-reference")]
    [string[]]$Modes = @("gpu-bf16", "gpu-f32"),
    [ValidateSet("chat-user", "prompt-text")]
    [string]$PromptMode = "chat-user",
    [string]$PromptName = "chat_short_joke",
    [string]$PromptText = "Tell me a short joke.",
    [int]$Runs = 3,
    [int]$WarmupRuns = 1,
    [int]$MaxNewTokens = 128,
    [int]$MaxContext = 256,
    [double]$Temperature = 0.0,
    [double]$TopP = 0.8,
    [int]$TopK = 20,
    [double]$RepeatPenalty = 1.05,
    [int64]$Seed = 123,
    [switch]$ProfileSync
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

function Invoke-BenchmarkRun {
    param(
        [Parameter(Mandatory = $true)][string]$ExePath,
        [Parameter(Mandatory = $true)][string]$Mode,
        [Parameter(Mandatory = $true)][string]$ModelDir,
        [Parameter(Mandatory = $true)][string]$PromptMode,
        [Parameter(Mandatory = $true)][string]$PromptText,
        [Parameter(Mandatory = $true)][int]$MaxNewTokens,
        [Parameter(Mandatory = $true)][int]$MaxContext,
        [Parameter(Mandatory = $true)][double]$Temperature,
        [Parameter(Mandatory = $true)][double]$TopP,
        [Parameter(Mandatory = $true)][int]$TopK,
        [Parameter(Mandatory = $true)][double]$RepeatPenalty,
        [Parameter(Mandatory = $true)][int64]$Seed,
        [Parameter(Mandatory = $true)][bool]$ProfileSyncEnabled,
        [Parameter(Mandatory = $true)][string]$ProfileJsonPath
    )

    $args = @()
    switch ($Mode) {
        "gpu-bf16" {
            $args += @("--infer-gpu", "--gpu-bf16")
        }
        "gpu-f32" {
            $args += @("--infer-gpu", "--gpu-f32-matvec")
        }
        "cpu-reference" {
            $args += @("--infer-reference")
        }
        default {
            throw "Unsupported mode: $Mode"
        }
    }

    $args += @(
        "--hf-model-dir", $ModelDir,
        "--max-new-tokens", "$MaxNewTokens",
        "--max-context", "$MaxContext",
        "--temperature", (To-InvariantString $Temperature),
        "--top-p", (To-InvariantString $TopP),
        "--top-k", "$TopK",
        "--repeat-penalty", (To-InvariantString $RepeatPenalty),
        "--seed", "$Seed",
        "--profile-json", $ProfileJsonPath
    )

    if ($PromptMode -eq "chat-user") {
        $args += @("--chat-user", $PromptText)
    } else {
        $args += @("--prompt-text", $PromptText)
    }

    if ($ProfileSyncEnabled -and $Mode -ne "cpu-reference") {
        $args += @("--profile-sync")
    }

    Write-Host ("Running mode={0} prompt={1}" -f $Mode, $PromptMode) -ForegroundColor Cyan
    $runOutput = & $ExePath @args 2>&1
    foreach ($line in $runOutput) {
        Write-Host $line
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark run failed (mode=$Mode, exit_code=$LASTEXITCODE)."
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

New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedCsvOut) -Force | Out-Null
New-Item -ItemType Directory -Path $profileTmpDir -Force | Out-Null

Write-Host ("Sequential benchmark start: modes={0}, warmup={1}, runs={2}" -f ($Modes -join ","), $WarmupRuns, $Runs) -ForegroundColor Green
Write-Host ("CSV output: {0}" -f $resolvedCsvOut) -ForegroundColor Green

foreach ($mode in $Modes) {
    for ($warm = 1; $warm -le $WarmupRuns; ++$warm) {
        $warmProfile = Join-Path $profileTmpDir ("warmup_{0}_{1}_{2}.json" -f $mode, $warm, [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())
        try {
            $null = Invoke-BenchmarkRun `
                -ExePath $resolvedExe `
                -Mode $mode `
                -ModelDir $resolvedModelDir `
                -PromptMode $PromptMode `
                -PromptText $PromptText `
                -MaxNewTokens $MaxNewTokens `
                -MaxContext $MaxContext `
                -Temperature $Temperature `
                -TopP $TopP `
                -TopK $TopK `
                -RepeatPenalty $RepeatPenalty `
                -Seed $Seed `
                -ProfileSyncEnabled $ProfileSync.IsPresent `
                -ProfileJsonPath $warmProfile
            Write-Host ("Warmup completed: mode={0} run={1}/{2}" -f $mode, $warm, $WarmupRuns) -ForegroundColor DarkGray
        } finally {
            if (Test-Path $warmProfile) {
                Remove-Item -LiteralPath $warmProfile -Force
            }
        }
    }

    for ($runIndex = 1; $runIndex -le $Runs; ++$runIndex) {
        $profilePath = Join-Path $profileTmpDir ("run_{0}_{1}_{2}.json" -f $mode, $runIndex, [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())
        try {
            $profile = Invoke-BenchmarkRun `
                -ExePath $resolvedExe `
                -Mode $mode `
                -ModelDir $resolvedModelDir `
                -PromptMode $PromptMode `
                -PromptText $PromptText `
                -MaxNewTokens $MaxNewTokens `
                -MaxContext $MaxContext `
                -Temperature $Temperature `
                -TopP $TopP `
                -TopK $TopK `
                -RepeatPenalty $RepeatPenalty `
                -Seed $Seed `
                -ProfileSyncEnabled $ProfileSync.IsPresent `
                -ProfileJsonPath $profilePath

            $row = [PSCustomObject]@{
                timestamp_utc    = [DateTime]::UtcNow.ToString("o")
                run_label        = $RunLabel
                mode             = $mode
                run_index        = $runIndex
                prompt_name      = $PromptName
                prompt_tokens    = [int]$profile.prompt_tokens
                generated_tokens = [int]$profile.generated_tokens
                load_time_ms     = To-InvariantString $profile.load_time_ms
                decode_time_ms   = To-InvariantString $profile.decode_time_ms
                tokens_per_second = To-InvariantString $profile.tokens_per_second
            }

            if (Test-Path $resolvedCsvOut) {
                $row | Export-Csv -LiteralPath $resolvedCsvOut -NoTypeInformation -Append
            } else {
                $row | Export-Csv -LiteralPath $resolvedCsvOut -NoTypeInformation
            }

            Write-Host ("Recorded: mode={0} run={1}/{2} tps={3}" -f $mode, $runIndex, $Runs, $row.tokens_per_second) -ForegroundColor Yellow
        } finally {
            if (Test-Path $profilePath) {
                Remove-Item -LiteralPath $profilePath -Force
            }
        }
    }
}

Write-Host "Benchmark complete." -ForegroundColor Green
Write-Host ("CSV written to: {0}" -f $resolvedCsvOut) -ForegroundColor Green
