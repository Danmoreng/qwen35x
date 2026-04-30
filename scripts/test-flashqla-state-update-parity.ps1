[CmdletBinding()]
param(
    [string]$Executable = "build/qwen35x_flashqla_state_update_parity.exe",
    [switch]$SkipBuild,
    [double]$Tolerance = 1.0e-6
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

Push-Location $RepoRoot
try {
    $resolvedExecutable = Resolve-RepoPath -Path $Executable -RepoRoot $RepoRoot

    Write-Host "FlashQLA state-update parity harness" -ForegroundColor Cyan
    Write-Host "Repo: $RepoRoot"
    Write-Host "Executable: $resolvedExecutable"
    Write-Host "Tolerance: $Tolerance"

    if (-not $SkipBuild) {
        Invoke-Step -Name "Build state-update parity target" -Action {
            & (Join-Path $ScriptDir "build.ps1") `
                -UseNinja `
                -EnableCuda `
                -Configuration Release `
                -Target qwen35x_flashqla_state_update_parity
        }
    }

    Invoke-Step -Name "State-update parity" -Action {
        & $resolvedExecutable --tolerance $Tolerance
    }

    Write-Host ""
    Write-Host "FlashQLA state-update parity complete." -ForegroundColor Green
} finally {
    Pop-Location
}
