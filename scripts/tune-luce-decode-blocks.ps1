[CmdletBinding()]
param(
    [string]$Executable = "build/qwen35x_lucebench.exe",
    [string]$HFModelDir = "models/qwen3.5-0.8b",
    [string]$SweepCsvOut = "benchmarks/luce-decode-blocks-sweep.csv",
    [string]$FinalCsvOut = "benchmarks/luce-decode-blocks-final.csv",
    [int]$MaxNewTokens = 128,
    [int]$MaxContext = 256
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
$resolvedSweepCsv = Resolve-RepoPath -Path $SweepCsvOut -RepoRoot $repoRoot
$resolvedFinalCsv = Resolve-RepoPath -Path $FinalCsvOut -RepoRoot $repoRoot

if (-not (Test-Path $resolvedExe)) {
    throw "Executable not found: $resolvedExe"
}
if (-not (Test-Path $resolvedModelDir)) {
    throw "Model directory not found: $resolvedModelDir"
}

New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedSweepCsv) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedFinalCsv) -Force | Out-Null

if (Test-Path $resolvedSweepCsv) { Remove-Item -LiteralPath $resolvedSweepCsv -Force }
if (Test-Path $resolvedFinalCsv) { Remove-Item -LiteralPath $resolvedFinalCsv -Force }

$queryOut = & $resolvedExe --query-decode-blocks 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to query max safe decode blocks."
}
$match = [regex]::Match(($queryOut -join "`n"), "max_safe_decode_blocks:\s*(\d+)")
if (-not $match.Success) {
    throw "Could not parse max_safe_decode_blocks from output."
}
$maxSafe = [int]$match.Groups[1].Value

$candidateSet = New-Object System.Collections.Generic.HashSet[int]
for ($b = 8; $b -le $maxSafe; $b += 8) {
    $null = $candidateSet.Add($b)
}
$null = $candidateSet.Add($maxSafe)
if ($maxSafe -gt 4) { $null = $candidateSet.Add($maxSafe - 4) }
if ($maxSafe -gt 8) { $null = $candidateSet.Add($maxSafe - 8) }
if ($maxSafe -gt 12) { $null = $candidateSet.Add($maxSafe - 12) }

$candidates = @($candidateSet) | Sort-Object
Write-Host ("Tuning decode blocks: max_safe={0}, candidates={1}" -f $maxSafe, ($candidates -join ",")) -ForegroundColor Cyan

foreach ($b in $candidates) {
    & (Join-Path $scriptDir "benchmark-luce-megakernel-seq.ps1") `
        -Executable $resolvedExe `
        -HFModelDir $resolvedModelDir `
        -CsvOut $resolvedSweepCsv `
        -RunLabel ("decode_blocks=" + $b) `
        -Runs 1 `
        -WarmupRuns 1 `
        -MaxNewTokens $MaxNewTokens `
        -MaxContext $MaxContext `
        -DecodeBlocks $b
    if ($LASTEXITCODE -ne 0) {
        throw "Sweep run failed for decode blocks = $b"
    }
}

$sweep = Import-Csv -LiteralPath $resolvedSweepCsv
if (-not $sweep -or $sweep.Count -eq 0) {
    throw "Sweep CSV has no rows: $resolvedSweepCsv"
}

$ranked = $sweep |
    Sort-Object @{Expression = { [double]$_.tokens_per_second }; Descending = $true}, @{Expression = { [int]$_.decode_blocks }; Descending = $false}
$top = $ranked | Select-Object -First 3

Write-Host "Top sweep candidates by decode tok/s:" -ForegroundColor Green
$top | Format-Table decode_blocks, tokens_per_second, pp_tokens_per_second -AutoSize | Out-String | Write-Host

foreach ($row in $top) {
    $b = [int]$row.decode_blocks
    & (Join-Path $scriptDir "benchmark-luce-megakernel-seq.ps1") `
        -Executable $resolvedExe `
        -HFModelDir $resolvedModelDir `
        -CsvOut $resolvedFinalCsv `
        -RunLabel ("validate_decode_blocks=" + $b) `
        -Runs 3 `
        -WarmupRuns 1 `
        -MaxNewTokens $MaxNewTokens `
        -MaxContext $MaxContext `
        -DecodeBlocks $b
    if ($LASTEXITCODE -ne 0) {
        throw "Validation run failed for decode blocks = $b"
    }
}

$final = Import-Csv -LiteralPath $resolvedFinalCsv
$summary = $final |
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
    Sort-Object mean_decode_tok_s -Descending

Write-Host "Validation summary:" -ForegroundColor Green
$summary | Format-Table -AutoSize | Out-String | Write-Host

$best = $summary | Select-Object -First 1
Write-Host ("Best decode blocks for this system: {0} (mean tg tok/s={1:N2})" -f $best.decode_blocks, [double]$best.mean_decode_tok_s) -ForegroundColor Yellow
Write-Host ("Sweep CSV: {0}" -f $resolvedSweepCsv) -ForegroundColor Green
Write-Host ("Validation CSV: {0}" -f $resolvedFinalCsv) -ForegroundColor Green
