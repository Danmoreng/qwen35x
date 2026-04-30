[CmdletBinding()]
param(
    [string]$Executable = "build/qwen35x.exe",
    [string]$CsvPrefix = "benchmarks/qwen35x-flashqla-prefill-parity",
    [string]$RunLabel = "flashqla-prefill-parity",
    [switch]$SkipBuild,
    [switch]$KeepProfiles
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent $ScriptDir
$previousFlashqlaTc = $env:QWEN35X_ENABLE_FLASHQLA_GDR_TC_PREFILL

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

Push-Location $RepoRoot
try {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $resolvedExecutable = Resolve-RepoPath -Path $Executable -RepoRoot $RepoRoot
    $csvOut = Resolve-RepoPath -Path "${CsvPrefix}-${timestamp}.csv" -RepoRoot $RepoRoot

    $env:QWEN35X_ENABLE_FLASHQLA_GDR_TC_PREFILL = "1"

    Write-Host "FlashQLA prefill parity harness" -ForegroundColor Cyan
    Write-Host "Repo: $RepoRoot"
    Write-Host "Executable: $resolvedExecutable"
    Write-Host "FlashQLA TC prefill: QWEN35X_ENABLE_FLASHQLA_GDR_TC_PREFILL=1"
    Write-Host "CSV: $csvOut"
    Write-Host "Mode: first-token CPU/GPU parity"

    if (-not $SkipBuild) {
        Invoke-Step -Name "Build qwen35x" -Action {
            & (Join-Path $ScriptDir "build.ps1") `
                -UseNinja `
                -EnableCuda `
                -Configuration Release `
                -Target qwen35x
        }
    }

    Invoke-Step -Name "First-token prefill parity" -Action {
        $parityArgs = @{
            Executable = $resolvedExecutable
            PromptsFile = "scripts/bench/parity_prompts_minimal.txt"
            CsvOut = $csvOut
            RunLabel = $RunLabel
            MaxNewTokens = 1
            MaxContext = 256
            GpuMode = "gpu-f32"
            Qwen35xPrefillMode = "batched"
        }
        if ($KeepProfiles.IsPresent) {
            $parityArgs.KeepProfiles = $true
        }
        & (Join-Path $ScriptDir "benchmark-parity.ps1") @parityArgs
    }

    Write-Host ""
    Write-Host "FlashQLA prefill parity complete." -ForegroundColor Green
    Write-Host "CSV: $csvOut"
} finally {
    if ($null -eq $previousFlashqlaTc) {
        Remove-Item Env:QWEN35X_ENABLE_FLASHQLA_GDR_TC_PREFILL -ErrorAction SilentlyContinue
    } else {
        $env:QWEN35X_ENABLE_FLASHQLA_GDR_TC_PREFILL = $previousFlashqlaTc
    }
    Pop-Location
}
