[CmdletBinding()]
param(
    [string]$Executable = "build-multi-request/qwen35x_cpu_engine_bench.exe",
    [string]$OutputDir = "benchmarks/multi-request-2026-09-06/static",
    [int[]]$BatchSizes = @(1, 2, 4, 8, 16, 32),
    [int[]]$PromptLengths = @(512, 4096),
    [int]$MaxNewTokens = 128,
    [int]$Threads = 8,
    [int]$Runs = 3,
    [int]$WarmupRuns = 1,
    [int64]$Affinity = 65535,
    [switch]$SerialReference,
    [switch]$GreedyTiles,
    [switch]$FullLogits,
    [switch]$ParallelStateBatches,
    [switch]$SerialStateBatches,
    [int]$RegisteredPrefixTokens = 0,
    [int]$SingleFlightPrefixTokens = 0,
    [switch]$IdenticalPrompts,
    [switch]$Serving,
    [int]$Residents = 16,
    [int]$DecodeBatchLimit = 16,
    [double]$ArrivalGapMs = 0
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (Test-Path -LiteralPath $OutputDir) { throw "Output directory already exists: $OutputDir" }
New-Item -ItemType Directory -Path $OutputDir | Out-Null
$fixture = (Get-Content configs/qwen3_5_0_8b_text_1024x512_tokens.csv -Raw).Trim().Split(',')
$process = [Diagnostics.Process]::GetCurrentProcess()
$previousAffinity = $process.ProcessorAffinity
@{
    revision = (git rev-parse HEAD)
    executable = $Executable
    executable_sha256 = (Get-FileHash -LiteralPath $Executable).Hash
    threads = $Threads
    affinity = $Affinity
    serial_reference = [bool]$SerialReference
    greedy_tiles = [bool]$GreedyTiles
    full_logits = [bool]$FullLogits
    parallel_state_batches = [bool]$ParallelStateBatches
    serial_state_batches = [bool]$SerialStateBatches
    registered_prefix_tokens = $RegisteredPrefixTokens
    single_flight_prefix_tokens = $SingleFlightPrefixTokens
    identical_prompts = [bool]$IdenticalPrompts
    serving = [bool]$Serving
    resident_limit = $Residents
    decode_batch_limit = $DecodeBatchLimit
    arrival_gap_ms = $ArrivalGapMs
    fixture = "configs/qwen3_5_0_8b_text_1024x512_tokens.csv"
    prompt_rule = "Repeat/truncate fixture; request r increments final prompt token by r modulo vocabulary."
    timing = "Decode after all first tokens; actual B*(N-1) forwards over common wall time including sampling and loop."
} | ConvertTo-Json | Set-Content -LiteralPath "$OutputDir/metadata.json"
try {
    $process.ProcessorAffinity = [IntPtr]$Affinity
    foreach ($length in $PromptLengths) {
        $prompt = (@(for ($i = 0; $i -lt $length; ++$i) { $fixture[$i % $fixture.Count] }) -join ',')
        foreach ($batch in $BatchSizes) {
            $label = "p$length-b$batch"
            $parameters = @{
                Executable = $Executable; Modes = @('cpu-h128')
                CpuQ4H128 = 'models/qwen3.5-0.8b/model-q4-h128-cpu-dot4.q35h'
                CpuThreads = $Threads; CpuBatch = $batch; CpuBatchSerial = $SerialReference
                CpuBatchGreedy = $GreedyTiles
                CpuBatchFullLogits = $FullLogits
                CpuBatchParallelState = $ParallelStateBatches
                CpuBatchSerialState = $SerialStateBatches
                CpuRegisteredPrefixTokens = $RegisteredPrefixTokens
                CpuSingleFlightPrefixTokens = $SingleFlightPrefixTokens
                CpuIdenticalPrompts = $IdenticalPrompts
                CpuServing = $Serving; CpuResidents = $Residents
                CpuDecodeBatchLimit = $DecodeBatchLimit; CpuArrivalGapMs = $ArrivalGapMs
                PromptMode = 'prompt-tokens'; PromptTokensCsv = $prompt; PromptName = $label
                MaxContext = $length + $MaxNewTokens; MaxNewTokens = $MaxNewTokens
                Runs = $Runs; WarmupRuns = $WarmupRuns
                CsvOut = "$OutputDir/$label.csv"; RunLabel = $label
                KeepProfiles = $true; ProfileDir = "$OutputDir/profiles"
            }
            & "$PSScriptRoot/benchmark-inference-seq.ps1" @parameters *> "$OutputDir/$label.log"
            if (@(Import-Csv -LiteralPath "$OutputDir/$label.csv").Count -ne $Runs) {
                throw "Incomplete benchmark $label"
            }
            Write-Host "Completed $label"
        }
    }
} finally { $process.ProcessorAffinity = $previousAffinity }
