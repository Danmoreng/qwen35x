[CmdletBinding()]
param(
    [string]$Executable = "build-multi-request/qwen35x_cpu_engine_bench.exe",
    [string]$OutputDir = "benchmarks/multi-request-2026-09-06/threads",
    [int[]]$BatchSizes = @(16),
    [int[]]$PromptLengths = @(512),
    [string[]]$Cases = @("cache96-t8", "cache96-t12", "cache96-t16", "all-t8", "all-t12", "all-t16", "physical-t16"),
    [int]$Runs = 3,
    [int]$WarmupRuns = 1
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (Test-Path -LiteralPath $OutputDir) { throw "Output directory exists: $OutputDir" }
New-Item -ItemType Directory -Path $OutputDir | Out-Null
# Ryzen 9 9955HX3D topology reported by GetLogicalProcessorInformationEx:
# paired SMT bits, CPUs 0..15 share 96 MiB L3; CPUs 16..31 share 32 MiB L3.
# These masks are machine-specific, not a portable topology-selection policy.
$configurations = @{
    "cache96-t8" = @(8, 65535)
    "cache96-t12" = @(12, 65535)
    "cache96-t16" = @(16, 65535)
    "all-t8" = @(8, 4294967295L)
    "all-t12" = @(12, 4294967295L)
    "all-t16" = @(16, 4294967295L)
    "physical-t12" = @(12, 1431655765L)
    "physical-t16" = @(16, 1431655765L)
    "all-t24" = @(24, 4294967295L)
    "all-t32" = @(32, 4294967295L)
}
foreach ($case in $Cases) {
    if (-not $configurations.ContainsKey($case)) { throw "Unknown topology case: $case" }
    $configuration = $configurations[$case]
    & "$PSScriptRoot/benchmark-cpu-batch.ps1" -Executable $Executable `
        -OutputDir "$OutputDir/$case" -BatchSizes $BatchSizes -PromptLengths $PromptLengths `
        -Threads $configuration[0] -Affinity $configuration[1] -Runs $Runs -WarmupRuns $WarmupRuns
    Write-Host "Completed topology case $case"
}
