[CmdletBinding()]
param(
    [string]$Executable = "build-multi-request/qwen35x_cpu_engine_bench.exe",
    [string]$OutputDir = "benchmarks/multi-request-2026-09-06/prefix",
    [int[]]$PromptLengths = @(512),
    [int]$BatchSize = 16,
    [int]$Threads = 8,
    [long]$Affinity = 65535,
    [int]$Runs = 3,
    [int]$WarmupRuns = 1,
    [switch]$FullHits
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (Test-Path -LiteralPath $OutputDir) { throw "Output directory exists: $OutputDir" }
New-Item -ItemType Directory -Path $OutputDir | Out-Null
foreach ($length in $PromptLengths) {
    if ($length -lt 2) { throw "Prefix comparison requires at least two prompt tokens." }
    $leg = 0
    foreach ($variant in @("cold", "warm", "warm", "cold")) {
        $prefixCount = if ($variant -eq "cold") { 0 } elseif ($FullHits) { $length } else { $length - 1 }
        & "$PSScriptRoot/benchmark-cpu-batch.ps1" -Executable $Executable `
            -OutputDir "$OutputDir/p$length-$leg-$variant" -BatchSizes $BatchSize `
            -PromptLengths $length -Threads $Threads -Affinity $Affinity `
            -RegisteredPrefixTokens $prefixCount -IdenticalPrompts:$FullHits `
            -Runs $Runs -WarmupRuns $WarmupRuns
        ++$leg
    }
}
