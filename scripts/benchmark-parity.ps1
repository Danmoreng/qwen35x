[CmdletBinding()]
param(
    [string]$Executable = "build/qwen35x.exe",
    [string]$HFModelDir = "models/qwen3.5-0.8b",
    [string]$PromptsFile = "scripts/bench/parity_prompts_minimal.txt",
    [string]$CsvOut = "benchmarks/qwen35x-parity.csv",
    [string]$RunLabel = "parity",
    [int]$MaxNewTokens = 4,
    [int]$MaxContext = 256,
    [double]$Temperature = 0.0,
    [double]$TopP = 0.8,
    [int]$TopK = 20,
    [double]$RepeatPenalty = 1.05,
    [int64]$Seed = 123,
    [ValidateSet("gpu-bf16", "gpu-f32")]
    [string]$GpuMode = "gpu-f32",
    [Alias("LucePrefillMode")]
    [ValidateSet("default", "replay", "batched")]
    [string]$Qwen35xPrefillMode = "default",
    [ValidateSet("bf16", "nvfp4")]
    [string]$Qwen35xWeightPrecision = "bf16",
    [ValidateSet("bf16", "quantized")]
    [string]$Qwen35xCachePrecision = "bf16",
    [switch]$ProfileSync,
    [bool]$FailOnMismatch = $true,
    [switch]$KeepProfiles
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

function Read-ParityPrompts {
    param([Parameter(Mandatory = $true)][string]$PromptsPath)

    $lineNumber = 0
    $prompts = New-Object System.Collections.Generic.List[object]
    $nameSet = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::OrdinalIgnoreCase)

    foreach ($line in Get-Content -LiteralPath $PromptsPath) {
        $lineNumber += 1
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith("#")) {
            continue
        }

        $prompt = $null
        try {
            $prompt = $line | ConvertFrom-Json -ErrorAction Stop
        } catch {
            throw "Invalid JSON prompt line at ${PromptsPath}:$lineNumber"
        }

        $name = [string]$prompt.name
        $mode = [string]$prompt.mode
        $text = [string]$prompt.text

        if ([string]::IsNullOrWhiteSpace($name)) {
            throw "Missing 'name' at ${PromptsPath}:$lineNumber"
        }
        if (-not $nameSet.Add($name)) {
            throw "Duplicate prompt name '$name' at ${PromptsPath}:$lineNumber"
        }
        if ($mode -ne "chat-user" -and $mode -ne "prompt-text") {
            throw "Unsupported prompt mode '$mode' at ${PromptsPath}:$lineNumber (allowed: chat-user, prompt-text)."
        }
        if ([string]::IsNullOrWhiteSpace($text)) {
            throw "Missing 'text' for prompt '$name' at ${PromptsPath}:$lineNumber"
        }

        $prompts.Add([PSCustomObject]@{
                name = $name
                mode = $mode
                text = $text
            })
    }

    return $prompts
}

function Invoke-InferenceProfile {
    param(
        [Parameter(Mandatory = $true)][string]$ExePath,
        [Parameter(Mandatory = $true)][string]$Mode,
        [Parameter(Mandatory = $true)][string]$ModelDir,
        [Parameter(Mandatory = $true)][string]$PromptMode,
        [Parameter(Mandatory = $true)][string]$PromptText,
        [Parameter(Mandatory = $true)][int]$MaxNewTokens,
        [Parameter(Mandatory = $true)][int]$MaxContext,
        [Parameter(Mandatory = $true)][double]$Temperature,
        [Parameter(Mandatory = $true)][double]$TopP,
        [Parameter(Mandatory = $true)][int]$TopK,
        [Parameter(Mandatory = $true)][double]$RepeatPenalty,
        [Parameter(Mandatory = $true)][int64]$Seed,
        [Parameter(Mandatory = $true)][string]$Qwen35xPrefillMode,
        [Parameter(Mandatory = $true)][string]$Qwen35xWeightPrecision,
        [Parameter(Mandatory = $true)][string]$Qwen35xCachePrecision,
        [Parameter(Mandatory = $true)][bool]$ProfileSyncEnabled,
        [Parameter(Mandatory = $true)][string]$ProfileJsonPath
    )

    $args = @()
    switch ($Mode) {
        "cpu-reference" {
            $args += @("--infer-reference")
        }
        "gpu-bf16" {
            $args += @("--infer-gpu", "--gpu-bf16")
        }
        "gpu-f32" {
            $args += @("--infer-gpu", "--gpu-f32-matvec")
        }
        default {
            throw "Unsupported mode: $Mode"
        }
    }

    $args += @(
        "--hf-model-dir", $ModelDir,
        "--max-new-tokens", "$MaxNewTokens",
        "--max-context", "$MaxContext",
        "--temperature", (To-InvariantString $Temperature),
        "--top-p", (To-InvariantString $TopP),
        "--top-k", "$TopK",
        "--repeat-penalty", (To-InvariantString $RepeatPenalty),
        "--seed", "$Seed",
        "--profile-json", $ProfileJsonPath
    )

    if ($PromptMode -eq "chat-user") {
        $args += @("--chat-user", $PromptText)
    } elseif ($PromptMode -eq "prompt-text") {
        $args += @("--prompt-text", $PromptText)
    } else {
        throw "Unsupported prompt mode: $PromptMode"
    }

    if ($ProfileSyncEnabled -and $Mode -ne "cpu-reference") {
        $args += @("--profile-sync")
    }
    if ($Mode -ne "cpu-reference" -and $Qwen35xPrefillMode -ne "default") {
        $args += @("--qwen35x-prefill-mode", $Qwen35xPrefillMode)
    }
    if ($Mode -ne "cpu-reference" -and $Qwen35xWeightPrecision -ne "bf16") {
        $args += @("--qwen35x-weight-precision", $Qwen35xWeightPrecision)
    }
    if ($Mode -ne "cpu-reference" -and $Qwen35xCachePrecision -ne "bf16") {
        $args += @("--qwen35x-cache-precision", $Qwen35xCachePrecision)
    }

    $runOutput = & $ExePath @args 2>&1
    foreach ($line in $runOutput) {
        Write-Host $line
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Inference run failed (mode=$Mode, exit_code=$LASTEXITCODE)."
    }
    if (-not (Test-Path -LiteralPath $ProfileJsonPath)) {
        throw "Missing profile JSON output: $ProfileJsonPath"
    }

    return Get-Content -Raw -LiteralPath $ProfileJsonPath | ConvertFrom-Json
}

function Compare-TokenSequences {
    param(
        [Parameter(Mandatory = $true)][int[]]$CpuTokens,
        [Parameter(Mandatory = $true)][int[]]$GpuTokens
    )

    $firstMismatchIndex = -1
    $cpuMismatchToken = ""
    $gpuMismatchToken = ""
    $sharedCount = [Math]::Min($CpuTokens.Count, $GpuTokens.Count)

    for ($i = 0; $i -lt $sharedCount; ++$i) {
        if ($CpuTokens[$i] -ne $GpuTokens[$i]) {
            $firstMismatchIndex = $i
            $cpuMismatchToken = [string]$CpuTokens[$i]
            $gpuMismatchToken = [string]$GpuTokens[$i]
            break
        }
    }

    if ($firstMismatchIndex -lt 0 -and $CpuTokens.Count -ne $GpuTokens.Count) {
        $firstMismatchIndex = $sharedCount
        if ($sharedCount -lt $CpuTokens.Count) {
            $cpuMismatchToken = [string]$CpuTokens[$sharedCount]
        }
        if ($sharedCount -lt $GpuTokens.Count) {
            $gpuMismatchToken = [string]$GpuTokens[$sharedCount]
        }
    }

    $isMatch = $firstMismatchIndex -lt 0
    return [PSCustomObject]@{
        is_match = $isMatch
        first_mismatch_index = $firstMismatchIndex
        cpu_mismatch_token = $cpuMismatchToken
        gpu_mismatch_token = $gpuMismatchToken
    }
}

$scriptDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $scriptDir

$resolvedExe = Resolve-RepoPath -Path $Executable -RepoRoot $repoRoot
$resolvedModelDir = Resolve-RepoPath -Path $HFModelDir -RepoRoot $repoRoot
$resolvedPrompts = Resolve-RepoPath -Path $PromptsFile -RepoRoot $repoRoot
$resolvedCsvOut = Resolve-RepoPath -Path $CsvOut -RepoRoot $repoRoot
$profileTmpDir = Join-Path $repoRoot "build\parity-profiles"

if (-not (Test-Path -LiteralPath $resolvedExe)) {
    throw "Executable not found: $resolvedExe"
}
if (-not (Test-Path -LiteralPath $resolvedModelDir)) {
    throw "Model directory not found: $resolvedModelDir"
}
if (-not (Test-Path -LiteralPath $resolvedPrompts)) {
    throw "Prompts file not found: $resolvedPrompts"
}

$prompts = Read-ParityPrompts -PromptsPath $resolvedPrompts
if ($prompts.Count -lt 1) {
    throw "No prompts found in $resolvedPrompts"
}

New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedCsvOut) -Force | Out-Null
New-Item -ItemType Directory -Path $profileTmpDir -Force | Out-Null

Write-Host ("Parity run start: prompts={0} gpu_mode={1}" -f $prompts.Count, $GpuMode) -ForegroundColor Green
Write-Host ("Prompts file: {0}" -f $resolvedPrompts) -ForegroundColor Green
Write-Host ("CSV output: {0}" -f $resolvedCsvOut) -ForegroundColor Green

$failures = New-Object System.Collections.Generic.List[string]
$promptIndex = 0

foreach ($prompt in $prompts) {
    $promptIndex += 1
    $safePromptName = ($prompt.name -replace "[^A-Za-z0-9_-]", "_")
    $timestampMs = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $cpuProfilePath = Join-Path $profileTmpDir ("cpu_{0:D2}_{1}_{2}.json" -f $promptIndex, $safePromptName, $timestampMs)
    $gpuProfilePath = Join-Path $profileTmpDir ("gpu_{0:D2}_{1}_{2}.json" -f $promptIndex, $safePromptName, $timestampMs)

    Write-Host ("[{0}/{1}] Prompt '{2}' mode={3}" -f $promptIndex, $prompts.Count, $prompt.name, $prompt.mode) -ForegroundColor Cyan

    try {
        $cpuProfile = Invoke-InferenceProfile `
            -ExePath $resolvedExe `
            -Mode "cpu-reference" `
            -ModelDir $resolvedModelDir `
            -PromptMode $prompt.mode `
            -PromptText $prompt.text `
            -MaxNewTokens $MaxNewTokens `
            -MaxContext $MaxContext `
            -Temperature $Temperature `
            -TopP $TopP `
            -TopK $TopK `
            -RepeatPenalty $RepeatPenalty `
            -Seed $Seed `
            -Qwen35xPrefillMode "default" `
            -Qwen35xWeightPrecision "bf16" `
            -Qwen35xCachePrecision "bf16" `
            -ProfileSyncEnabled $false `
            -ProfileJsonPath $cpuProfilePath

        $gpuProfile = Invoke-InferenceProfile `
            -ExePath $resolvedExe `
            -Mode $GpuMode `
            -ModelDir $resolvedModelDir `
            -PromptMode $prompt.mode `
            -PromptText $prompt.text `
            -MaxNewTokens $MaxNewTokens `
            -MaxContext $MaxContext `
            -Temperature $Temperature `
            -TopP $TopP `
            -TopK $TopK `
            -RepeatPenalty $RepeatPenalty `
            -Seed $Seed `
            -Qwen35xPrefillMode $Qwen35xPrefillMode `
            -Qwen35xWeightPrecision $Qwen35xWeightPrecision `
            -Qwen35xCachePrecision $Qwen35xCachePrecision `
            -ProfileSyncEnabled $ProfileSync.IsPresent `
            -ProfileJsonPath $gpuProfilePath

        $cpuTokens = @()
        if ($null -ne $cpuProfile.output_token_ids) {
            $cpuTokens = @($cpuProfile.output_token_ids | ForEach-Object { [int]$_ })
        }
        $gpuTokens = @()
        if ($null -ne $gpuProfile.output_token_ids) {
            $gpuTokens = @($gpuProfile.output_token_ids | ForEach-Object { [int]$_ })
        }

        $comparison = Compare-TokenSequences -CpuTokens $cpuTokens -GpuTokens $gpuTokens
        $parityPass = [bool]$comparison.is_match

        if ($parityPass) {
            Write-Host ("  parity=PASS generated_tokens={0}" -f $cpuTokens.Count) -ForegroundColor Green
        } else {
            Write-Host ("  parity=FAIL first_mismatch_index={0} cpu={1} gpu={2}" -f $comparison.first_mismatch_index, $comparison.cpu_mismatch_token, $comparison.gpu_mismatch_token) -ForegroundColor Red
            $failures.Add($prompt.name)
        }

        $row = [PSCustomObject]@{
            timestamp_utc = [DateTime]::UtcNow.ToString("o")
            run_label = $RunLabel
            prompt_index = $promptIndex
            prompt_name = $prompt.name
            prompt_mode = $prompt.mode
            prompt_text = $prompt.text
            gpu_mode = $GpuMode
            max_new_tokens = $MaxNewTokens
            max_context = $MaxContext
            temperature = To-InvariantString $Temperature
            top_p = To-InvariantString $TopP
            top_k = $TopK
            repeat_penalty = To-InvariantString $RepeatPenalty
            seed = $Seed
            qwen35x_prefill_mode = $Qwen35xPrefillMode
            qwen35x_weight_precision = $Qwen35xWeightPrecision
            qwen35x_cache_precision = $Qwen35xCachePrecision
            token_parity_pass = if ($parityPass) { "true" } else { "false" }
            first_mismatch_index = $comparison.first_mismatch_index
            cpu_mismatch_token = $comparison.cpu_mismatch_token
            gpu_mismatch_token = $comparison.gpu_mismatch_token
            cpu_generated_tokens = $cpuTokens.Count
            gpu_generated_tokens = $gpuTokens.Count
            cpu_tokens_per_second = To-InvariantString $cpuProfile.tokens_per_second
            gpu_tokens_per_second = To-InvariantString $gpuProfile.tokens_per_second
            cpu_profile_json = $cpuProfilePath
            gpu_profile_json = $gpuProfilePath
        }

        if (Test-Path -LiteralPath $resolvedCsvOut) {
            $row | Export-Csv -LiteralPath $resolvedCsvOut -NoTypeInformation -Append
        } else {
            $row | Export-Csv -LiteralPath $resolvedCsvOut -NoTypeInformation
        }
    } finally {
        if (-not $KeepProfiles.IsPresent) {
            if (Test-Path -LiteralPath $cpuProfilePath) {
                Remove-Item -LiteralPath $cpuProfilePath -Force
            }
            if (Test-Path -LiteralPath $gpuProfilePath) {
                Remove-Item -LiteralPath $gpuProfilePath -Force
            }
        }
    }
}

$passCount = $prompts.Count - $failures.Count
Write-Host ("Parity run complete: pass={0} fail={1}" -f $passCount, $failures.Count) -ForegroundColor Green
Write-Host ("CSV written to: {0}" -f $resolvedCsvOut) -ForegroundColor Green

if ($failures.Count -gt 0) {
    Write-Host ("Failed prompts: {0}" -f ($failures -join ", ")) -ForegroundColor Yellow
    if ($FailOnMismatch) {
        throw "Parity mismatches detected."
    }
}
