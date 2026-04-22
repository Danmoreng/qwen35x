[CmdletBinding()]
param(
    [string]$CustomCsvOut = "benchmarks/qwen35x-inference-seq-compare.csv",
    [string]$TinygradCsvOut = "benchmarks/tinygrad-inference-seq-compare.csv",
    [string]$LlamaJsonOut = "benchmarks/llama-bench/qwen3.5-0.8b-bf16-compare.json",
    [string]$SummaryCsvOut = "benchmarks/benchmark-compare-summary.csv",
    [string]$RunLabel = "compare",
    [int]$Runs = 3,
    [int]$WarmupRuns = 1,
    [int]$MaxNewTokens = 128,
    [int]$MaxContext = 256,
    [string]$PromptText = "Tell me a short joke.",
    [ValidateSet("gpu-bf16", "gpu-f32", "cpu-reference")]
    [string[]]$CustomModes = @("gpu-bf16"),
    [ValidateSet("CUDA:PTX", "CUDA", "CL", "CPU")]
    [string]$TinygradDevice = "CUDA:PTX",
    [int]$LlamaRepetitions = 3,
    [switch]$SkipCustom,
    [switch]$SkipTinygrad,
    [switch]$SkipLlama,
    [bool]$LlamaSkipConvert = $true,
    [bool]$LlamaSkipBuild = $true
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

$scriptDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $scriptDir

$resolvedCustomCsvOut = Resolve-RepoPath -Path $CustomCsvOut -RepoRoot $repoRoot
$resolvedTinygradCsvOut = Resolve-RepoPath -Path $TinygradCsvOut -RepoRoot $repoRoot
$resolvedLlamaJsonOut = Resolve-RepoPath -Path $LlamaJsonOut -RepoRoot $repoRoot
$resolvedSummaryCsvOut = Resolve-RepoPath -Path $SummaryCsvOut -RepoRoot $repoRoot

New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedCustomCsvOut) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedTinygradCsvOut) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedLlamaJsonOut) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedSummaryCsvOut) -Force | Out-Null

if (-not $SkipCustom) {
    Write-Host "Running custom qwen35x sequential benchmark..." -ForegroundColor Cyan
    & (Join-Path $scriptDir "benchmark-inference-seq.ps1") `
        -CsvOut $resolvedCustomCsvOut `
        -RunLabel $RunLabel `
        -Modes $CustomModes `
        -PromptText $PromptText `
        -Runs $Runs `
        -WarmupRuns $WarmupRuns `
        -MaxNewTokens $MaxNewTokens `
        -MaxContext $MaxContext
    if ($LASTEXITCODE -ne 0) {
        throw "Custom benchmark failed."
    }
}

if (-not $SkipTinygrad) {
    Write-Host "Running tinygrad sequential benchmark..." -ForegroundColor Cyan
    & (Join-Path $scriptDir "benchmark-tinygrad-seq.ps1") `
        -CsvOut $resolvedTinygradCsvOut `
        -RunLabel $RunLabel `
        -Device $TinygradDevice `
        -PromptText $PromptText `
        -Runs $Runs `
        -WarmupRuns $WarmupRuns `
        -MaxNewTokens $MaxNewTokens `
        -MaxContext $MaxContext
    if ($LASTEXITCODE -ne 0) {
        throw "tinygrad benchmark failed."
    }
}

if (-not $SkipLlama) {
    Write-Host "Running llama.cpp benchmark..." -ForegroundColor Cyan
    $llamaArgs = @(
        "-BenchOut", $resolvedLlamaJsonOut,
        "-PromptTokens", "$MaxContext",
        "-GenTokens", "$MaxNewTokens",
        "-Repetitions", "$LlamaRepetitions"
    )
    if ($LlamaSkipConvert) {
        $llamaArgs += "-SkipConvert"
    }
    if ($LlamaSkipBuild) {
        $llamaArgs += "-SkipBuild"
    }
    & (Join-Path $scriptDir "benchmark-llama-bf16.ps1") @llamaArgs
    if ($LASTEXITCODE -ne 0) {
        throw "llama.cpp benchmark failed."
    }
}

$rows = @()

if (Test-Path $resolvedCustomCsvOut) {
    $customRows = Import-Csv -LiteralPath $resolvedCustomCsvOut
    $customByMode = $customRows | Group-Object mode
    foreach ($group in $customByMode) {
        $avg = ($group.Group | Measure-Object -Property tokens_per_second -Average).Average
        $rows += [PSCustomObject]@{
            implementation     = "qwen35x"
            backend            = $group.Name
            avg_tokens_per_sec = To-InvariantString $avg
            source_file        = $resolvedCustomCsvOut
            run_label          = $RunLabel
        }
    }
}

if (Test-Path $resolvedTinygradCsvOut) {
    $tinyRows = Import-Csv -LiteralPath $resolvedTinygradCsvOut
    $avg = ($tinyRows | Measure-Object -Property tokens_per_second -Average).Average
    $backend = if ($tinyRows.Count -gt 0) { $tinyRows[0].backend } else { $TinygradDevice }
    $rows += [PSCustomObject]@{
        implementation     = "tinygrad"
        backend            = $backend
        avg_tokens_per_sec = To-InvariantString $avg
        source_file        = $resolvedTinygradCsvOut
        run_label          = $RunLabel
    }
}

if (Test-Path $resolvedLlamaJsonOut) {
    $llamaJson = Get-Content -Raw -LiteralPath $resolvedLlamaJsonOut | ConvertFrom-Json
    $genRows = @($llamaJson | Where-Object { $_.n_gen -gt 0 })
    if ($genRows.Count -gt 0) {
        $avg = ($genRows | Measure-Object -Property avg_ts -Average).Average
        $backend = [string]$genRows[0].backends
        $rows += [PSCustomObject]@{
            implementation     = "llama.cpp"
            backend            = $backend
            avg_tokens_per_sec = To-InvariantString $avg
            source_file        = $resolvedLlamaJsonOut
            run_label          = $RunLabel
        }
    }
}

if ($rows.Count -eq 0) {
    throw "No benchmark outputs were found to summarize."
}

$rows | Export-Csv -LiteralPath $resolvedSummaryCsvOut -NoTypeInformation
Write-Host "Comparison summary written to: $resolvedSummaryCsvOut" -ForegroundColor Green
$rows | Format-Table -AutoSize
