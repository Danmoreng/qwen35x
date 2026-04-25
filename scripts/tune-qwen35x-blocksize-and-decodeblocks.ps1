[CmdletBinding()]
param(
    [string]$Executable = "build/qwen35x_kernelbench.exe",
    [string]$HFModelDir = "models/qwen3.5-0.8b",
    [int[]]$BlockSizes = @(256, 384, 512),
    [int]$MaxNewTokens = 128,
    [int]$MaxContext = 256,
    [string]$SweepDir = "benchmarks/qwen35x-kernel-tuning",
    [string]$SummaryCsvOut = "benchmarks/qwen35x-kernel-tuning-summary.csv"
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

$scriptDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $scriptDir

$resolvedExe = Resolve-RepoPath -Path $Executable -RepoRoot $repoRoot
$resolvedModelDir = Resolve-RepoPath -Path $HFModelDir -RepoRoot $repoRoot
$resolvedSweepDir = Resolve-RepoPath -Path $SweepDir -RepoRoot $repoRoot
$resolvedSummaryCsv = Resolve-RepoPath -Path $SummaryCsvOut -RepoRoot $repoRoot

if (-not (Test-Path $resolvedModelDir)) {
    throw "Model directory not found: $resolvedModelDir"
}
if (-not $BlockSizes -or $BlockSizes.Count -eq 0) {
    throw "Provide at least one block size."
}

New-Item -ItemType Directory -Path $resolvedSweepDir -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedSummaryCsv) -Force | Out-Null
if (Test-Path $resolvedSummaryCsv) {
    Remove-Item -LiteralPath $resolvedSummaryCsv -Force
}

$summaryRows = @()

foreach ($bs in $BlockSizes) {
    if ($bs -lt 128 -or ($bs % 32) -ne 0) {
        throw "Invalid block size: $bs (must be >=128 and divisible by 32)."
    }

    Write-Host ("=== Tuning BLOCK_SIZE={0} ===" -f $bs) -ForegroundColor Cyan

    & (Join-Path $scriptDir "build.ps1") `
        -UseNinja `
        -EnableCuda `
        -Configuration Release `
        -Target qwen35x_kernelbench `
        -KernelBlockSize $bs
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed for BLOCK_SIZE=$bs"
    }

    $sweepCsv = Join-Path $resolvedSweepDir ("sweep_bs{0}.csv" -f $bs)
    $finalCsv = Join-Path $resolvedSweepDir ("final_bs{0}.csv" -f $bs)
    if (Test-Path $sweepCsv) { Remove-Item -LiteralPath $sweepCsv -Force }
    if (Test-Path $finalCsv) { Remove-Item -LiteralPath $finalCsv -Force }

    & (Join-Path $scriptDir "tune-qwen35x-decode-blocks.ps1") `
        -Executable $resolvedExe `
        -HFModelDir $resolvedModelDir `
        -SweepCsvOut $sweepCsv `
        -FinalCsvOut $finalCsv `
        -MaxNewTokens $MaxNewTokens `
        -MaxContext $MaxContext
    if ($LASTEXITCODE -ne 0) {
        throw "Decode-block tuning failed for BLOCK_SIZE=$bs"
    }

    $finalRows = Import-Csv -LiteralPath $finalCsv
    if (-not $finalRows -or $finalRows.Count -eq 0) {
        throw "No final tuning rows for BLOCK_SIZE=$bs"
    }

    $bestGroup = $finalRows |
        Group-Object decode_blocks |
        ForEach-Object {
            $vals = $_.Group | ForEach-Object { [double]$_.tokens_per_second }
            [PSCustomObject]@{
                decode_blocks = [int]$_.Name
                mean_decode_tok_s = ($vals | Measure-Object -Average).Average
                min_decode_tok_s = ($vals | Measure-Object -Minimum).Minimum
                max_decode_tok_s = ($vals | Measure-Object -Maximum).Maximum
            }
        } |
        Sort-Object mean_decode_tok_s -Descending |
        Select-Object -First 1

    $summaryRows += [PSCustomObject]@{
        block_size = $bs
        best_decode_blocks = [int]$bestGroup.decode_blocks
        mean_decode_tok_s = [double]$bestGroup.mean_decode_tok_s
        min_decode_tok_s = [double]$bestGroup.min_decode_tok_s
        max_decode_tok_s = [double]$bestGroup.max_decode_tok_s
        final_csv = $finalCsv
        sweep_csv = $sweepCsv
    }
}

$summaryRows | Sort-Object mean_decode_tok_s -Descending | Export-Csv -NoTypeInformation -LiteralPath $resolvedSummaryCsv

Write-Host "=== Block size tuning summary ===" -ForegroundColor Green
($summaryRows | Sort-Object mean_decode_tok_s -Descending | Format-Table -AutoSize | Out-String) | Write-Host
Write-Host ("Summary CSV: {0}" -f $resolvedSummaryCsv) -ForegroundColor Green

$bestOverall = $summaryRows | Sort-Object mean_decode_tok_s -Descending | Select-Object -First 1
Write-Host ("Best overall config: BLOCK_SIZE={0}, decode_blocks={1}, mean tg tok/s={2:N2}" -f `
    $bestOverall.block_size, $bestOverall.best_decode_blocks, [double]$bestOverall.mean_decode_tok_s) -ForegroundColor Yellow
