[CmdletBinding()]
param(
    [string]$Baseline = "benchmarks/multi-request-2026-09-06/static-full-logits.exe",
    [string]$Candidate = "build-multi-request/qwen35x_cpu_engine_bench.exe",
    [string]$OutputDir = "benchmarks/multi-request-2026-09-06/batch-pair",
    [int[]]$BatchSizes = @(4, 16),
    [int[]]$PromptLengths = @(512),
    [int]$BaseThreads = 8,
    [int]$CandidateThreads = 8,
    [long]$BaseAffinity = 65535,
    [long]$CandidateAffinity = 65535,
    [switch]$CandidateGreedyTiles,
    [switch]$BaselineFullLogits,
    [int]$Runs = 3,
    [int]$WarmupRuns = 1
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (Test-Path -LiteralPath $OutputDir) { throw "Output directory exists: $OutputDir" }
New-Item -ItemType Directory -Path $OutputDir | Out-Null
foreach ($length in $PromptLengths) {
    foreach ($batch in $BatchSizes) {
        $leg = 0
        foreach ($variant in @("base", "candidate", "candidate", "base")) {
            $isCandidate = $variant -eq "candidate"
            $parameters = @{
                Executable = $(if ($isCandidate) { $Candidate } else { $Baseline })
                Threads = $(if ($isCandidate) { $CandidateThreads } else { $BaseThreads })
                Affinity = $(if ($isCandidate) { $CandidateAffinity } else { $BaseAffinity })
                GreedyTiles = $isCandidate -and $CandidateGreedyTiles
                FullLogits = (-not $isCandidate) -and $BaselineFullLogits
                BatchSizes = @($batch); PromptLengths = @($length)
                OutputDir = "$OutputDir/p$length-b$batch-$leg-$variant"
                Runs = $Runs; WarmupRuns = $WarmupRuns
            }
            & "$PSScriptRoot/benchmark-cpu-batch.ps1" @parameters
            ++$leg
        }
    }
}
