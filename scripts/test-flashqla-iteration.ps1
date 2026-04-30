[CmdletBinding()]
param(
    [string]$Executable = "build/qwen35x.exe",
    [string]$CsvPrefix = "benchmarks/qwen35x-flashqla-iteration",
    [string]$RunLabel = "flashqla-iteration",
    [switch]$SkipBuild,
    [switch]$SkipParity,
    [switch]$SkipBenchmark
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent $ScriptDir

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

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    Write-Host ""
    Write-Host "== $Name ==" -ForegroundColor Cyan
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    & $Action
    $timer.Stop()
    Write-Host ("Completed {0} in {1:n1}s" -f $Name, $timer.Elapsed.TotalSeconds) -ForegroundColor Green
}

$previousFlashqlaTc = $env:QWEN35X_ENABLE_FLASHQLA_GDR_TC_PREFILL

Push-Location $RepoRoot
try {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $resolvedExecutable = Resolve-RepoPath -Path $Executable -RepoRoot $RepoRoot
    $parityCsv = Resolve-RepoPath -Path "${CsvPrefix}-minimal-parity-${timestamp}.csv" -RepoRoot $RepoRoot
    $benchmarkCsv = Resolve-RepoPath -Path "${CsvPrefix}-wiki-ai-64k-gen128-${timestamp}.csv" -RepoRoot $RepoRoot

    $env:QWEN35X_ENABLE_FLASHQLA_GDR_TC_PREFILL = "1"

    Write-Host "FlashQLA iteration harness" -ForegroundColor Cyan
    Write-Host "Repo: $RepoRoot"
    Write-Host "Executable: $resolvedExecutable"
    Write-Host "FlashQLA TC prefill: QWEN35X_ENABLE_FLASHQLA_GDR_TC_PREFILL=1"
    Write-Host "Parity CSV: $parityCsv"
    Write-Host "64k profile CSV: $benchmarkCsv"

    if (-not $SkipBuild) {
        Invoke-Step -Name "Build qwen35x" -Action {
            & (Join-Path $ScriptDir "build.ps1") `
                -UseNinja `
                -EnableCuda `
                -Configuration Release `
                -Target qwen35x
        }
    }

    if (-not $SkipParity) {
        Invoke-Step -Name "Minimal parity" -Action {
            & (Join-Path $ScriptDir "benchmark-parity.ps1") `
                -Executable $resolvedExecutable `
                -PromptsFile "scripts/bench/parity_prompts_minimal.txt" `
                -CsvOut $parityCsv `
                -RunLabel "${RunLabel}-minimal-parity" `
                -MaxNewTokens 4 `
                -MaxContext 256 `
                -GpuMode gpu-f32 `
                -Qwen35xPrefillMode batched
        }
    }

    if (-not $SkipBenchmark) {
        Invoke-Step -Name "64k Wikipedia profile benchmark" -Action {
            & (Join-Path $ScriptDir "benchmark-inference-seq.ps1") `
                -Executable $resolvedExecutable `
                -CsvOut $benchmarkCsv `
                -RunLabel "${RunLabel}-wiki-ai-64k-gen128" `
                -Modes gpu-f32 `
                -PromptMode prompt-file `
                -PromptFile "benchmarks/inputs/wiki_artificial_intelligence_64k_prompt.txt" `
                -PromptName wiki_ai_64k_gen128 `
                -Runs 3 `
                -WarmupRuns 1 `
                -MaxNewTokens 128 `
                -MaxContext 65536 `
                -Qwen35xPrefillMode batched `
                -Qwen35xProfile
        }
    }

    Write-Host ""
    Write-Host "FlashQLA iteration complete." -ForegroundColor Green
    Write-Host "Parity CSV: $parityCsv"
    Write-Host "64k profile CSV: $benchmarkCsv"
} finally {
    if ($null -eq $previousFlashqlaTc) {
        Remove-Item Env:QWEN35X_ENABLE_FLASHQLA_GDR_TC_PREFILL -ErrorAction SilentlyContinue
    } else {
        $env:QWEN35X_ENABLE_FLASHQLA_GDR_TC_PREFILL = $previousFlashqlaTc
    }
    Pop-Location
}
