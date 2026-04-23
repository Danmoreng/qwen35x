[CmdletBinding()]
param(
    [string]$Executable = "build/qwen35x.exe",
    [string]$PythonExe = ".venv-hf-parity\Scripts\python.exe",
    [string]$RunnerScript = "scripts/hf/transformers_inference.py",
    [string]$HFModelDir = "models/qwen3.5-0.8b",
    [string]$PromptsFile = "scripts/bench/parity_prompts_minimal.txt",
    [string]$CsvOut = "benchmarks/qwen35x-transformers-parity.csv",
    [string]$RunLabel = "transformers-parity",
    [int]$MaxNewTokens = 4,
    [int]$MaxContext = 256,
    [double]$Temperature = 0.0,
    [double]$TopP = 0.8,
    [int]$TopK = 20,
    [double]$RepeatPenalty = 1.05,
    [int64]$Seed = 123,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$HFDevice = "auto",
    [ValidateSet("auto", "float32", "bfloat16", "float16")]
    [string]$HFDtype = "float32",
    [ValidateSet("auto", "causal-lm", "image-text-to-text")]
    [string]$HFModelAutoClass = "auto",
    [switch]$TrustRemoteCode,
    [switch]$AllowDownload,
    [switch]$NoCache,
    [switch]$AllowMismatch,
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

function Resolve-ExecutableOrRepoPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$RepoRoot
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }

    $repoPath = Join-Path $RepoRoot $Path
    if (Test-Path -LiteralPath $repoPath) {
        return $repoPath
    }

    $command = Get-Command $Path -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    return $repoPath
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

function Compare-IntSequences {
    param(
        [Parameter(Mandatory = $true)][int[]]$Expected,
        [Parameter(Mandatory = $true)][int[]]$Actual
    )

    $firstMismatchIndex = -1
    $expectedMismatch = ""
    $actualMismatch = ""
    $sharedCount = [Math]::Min($Expected.Count, $Actual.Count)
    for ($i = 0; $i -lt $sharedCount; ++$i) {
        if ($Expected[$i] -ne $Actual[$i]) {
            $firstMismatchIndex = $i
            $expectedMismatch = [string]$Expected[$i]
            $actualMismatch = [string]$Actual[$i]
            break
        }
    }
    if ($firstMismatchIndex -lt 0 -and $Expected.Count -ne $Actual.Count) {
        $firstMismatchIndex = $sharedCount
        if ($sharedCount -lt $Expected.Count) {
            $expectedMismatch = [string]$Expected[$sharedCount]
        }
        if ($sharedCount -lt $Actual.Count) {
            $actualMismatch = [string]$Actual[$sharedCount]
        }
    }

    return [PSCustomObject]@{
        is_match = ($firstMismatchIndex -lt 0)
        first_mismatch_index = $firstMismatchIndex
        expected_mismatch_token = $expectedMismatch
        actual_mismatch_token = $actualMismatch
    }
}

function Invoke-QwenProfile {
    param(
        [Parameter(Mandatory = $true)][string]$ExePath,
        [Parameter(Mandatory = $true)][string]$ModelDir,
        [Parameter(Mandatory = $true)][string]$PromptMode,
        [Parameter(Mandatory = $true)][string]$PromptText,
        [Parameter(Mandatory = $true)][string]$ProfileJsonPath
    )

    $args = @(
        "--infer-reference",
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
    } else {
        $args += @("--prompt-text", $PromptText)
    }

    $runOutput = & $ExePath @args 2>&1
    foreach ($line in $runOutput) {
        Write-Host $line
    }
    if ($LASTEXITCODE -ne 0) {
        throw "qwen35x CPU reference failed (exit_code=$LASTEXITCODE)."
    }
    if (-not (Test-Path -LiteralPath $ProfileJsonPath)) {
        throw "Missing qwen35x profile JSON: $ProfileJsonPath"
    }
    return Get-Content -Raw -LiteralPath $ProfileJsonPath | ConvertFrom-Json
}

function Invoke-HFProfile {
    param(
        [Parameter(Mandatory = $true)][string]$ResolvedPython,
        [Parameter(Mandatory = $true)][string]$ResolvedRunner,
        [Parameter(Mandatory = $true)][string]$ModelDir,
        [Parameter(Mandatory = $true)][string]$PromptMode,
        [Parameter(Mandatory = $true)][string]$PromptText,
        [Parameter(Mandatory = $true)][string]$ProfileJsonPath
    )

    $args = @(
        $ResolvedRunner,
        "--model-dir", $ModelDir,
        "--prompt-mode", $PromptMode,
        "--prompt-text", $PromptText,
        "--max-new-tokens", "$MaxNewTokens",
        "--max-context", "$MaxContext",
        "--temperature", (To-InvariantString $Temperature),
        "--top-p", (To-InvariantString $TopP),
        "--top-k", "$TopK",
        "--repeat-penalty", (To-InvariantString $RepeatPenalty),
        "--seed", "$Seed",
        "--device", $HFDevice,
        "--dtype", $HFDtype,
        "--model-auto-class", $HFModelAutoClass,
        "--output-json", $ProfileJsonPath
    )
    if ($TrustRemoteCode.IsPresent) {
        $args += "--trust-remote-code"
    }
    if ($AllowDownload.IsPresent) {
        $args += "--allow-download"
    }
    if ($NoCache.IsPresent) {
        $args += "--no-cache"
    }

    $runOutput = & $ResolvedPython @args 2>&1
    foreach ($line in $runOutput) {
        Write-Host $line
    }
    if ($LASTEXITCODE -ne 0) {
        throw "HF Transformers parity runner failed (exit_code=$LASTEXITCODE)."
    }
    if (-not (Test-Path -LiteralPath $ProfileJsonPath)) {
        throw "Missing HF profile JSON: $ProfileJsonPath"
    }
    return Get-Content -Raw -LiteralPath $ProfileJsonPath | ConvertFrom-Json
}

$scriptDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $scriptDir

$resolvedExe = Resolve-RepoPath -Path $Executable -RepoRoot $repoRoot
$resolvedPython = Resolve-ExecutableOrRepoPath -Path $PythonExe -RepoRoot $repoRoot
$resolvedRunner = Resolve-RepoPath -Path $RunnerScript -RepoRoot $repoRoot
$resolvedModelDir = Resolve-RepoPath -Path $HFModelDir -RepoRoot $repoRoot
$resolvedPrompts = Resolve-RepoPath -Path $PromptsFile -RepoRoot $repoRoot
$resolvedCsvOut = Resolve-RepoPath -Path $CsvOut -RepoRoot $repoRoot
$profileTmpDir = Join-Path $repoRoot "build\transformers-parity-profiles"

if (-not (Test-Path -LiteralPath $resolvedExe)) {
    throw "Executable not found: $resolvedExe"
}
if (-not (Test-Path -LiteralPath $resolvedPython)) {
    throw "Python executable not found: $resolvedPython. Run scripts/setup-transformers-parity.ps1 or pass -PythonExe."
}
if (-not (Test-Path -LiteralPath $resolvedRunner)) {
    throw "HF runner not found: $resolvedRunner"
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

Write-Host ("Transformers parity start: prompts={0} device={1} dtype={2}" -f $prompts.Count, $HFDevice, $HFDtype) -ForegroundColor Green
Write-Host ("HF model auto class: {0}" -f $HFModelAutoClass) -ForegroundColor Green
Write-Host ("Prompts file: {0}" -f $resolvedPrompts) -ForegroundColor Green
Write-Host ("CSV output: {0}" -f $resolvedCsvOut) -ForegroundColor Green

$failures = New-Object System.Collections.Generic.List[string]
$promptIndex = 0

foreach ($prompt in $prompts) {
    $promptIndex += 1
    $safePromptName = ($prompt.name -replace "[^A-Za-z0-9_-]", "_")
    $timestampMs = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $qwenProfilePath = Join-Path $profileTmpDir ("qwen_cpu_{0:D2}_{1}_{2}.json" -f $promptIndex, $safePromptName, $timestampMs)
    $hfProfilePath = Join-Path $profileTmpDir ("hf_{0:D2}_{1}_{2}.json" -f $promptIndex, $safePromptName, $timestampMs)

    Write-Host ("[{0}/{1}] Prompt '{2}' mode={3}" -f $promptIndex, $prompts.Count, $prompt.name, $prompt.mode) -ForegroundColor Cyan

    try {
        $qwenProfile = Invoke-QwenProfile `
            -ExePath $resolvedExe `
            -ModelDir $resolvedModelDir `
            -PromptMode $prompt.mode `
            -PromptText $prompt.text `
            -ProfileJsonPath $qwenProfilePath

        $hfProfile = Invoke-HFProfile `
            -ResolvedPython $resolvedPython `
            -ResolvedRunner $resolvedRunner `
            -ModelDir $resolvedModelDir `
            -PromptMode $prompt.mode `
            -PromptText $prompt.text `
            -ProfileJsonPath $hfProfilePath

        $qwenPromptTokens = @()
        if ($null -ne $qwenProfile.prompt_token_ids) {
            $qwenPromptTokens = @($qwenProfile.prompt_token_ids | ForEach-Object { [int]$_ })
        }
        $hfPromptTokens = @()
        if ($null -ne $hfProfile.prompt_token_ids) {
            $hfPromptTokens = @($hfProfile.prompt_token_ids | ForEach-Object { [int]$_ })
        }
        $qwenTokens = @()
        if ($null -ne $qwenProfile.output_token_ids) {
            $qwenTokens = @($qwenProfile.output_token_ids | ForEach-Object { [int]$_ })
        }
        $hfTokens = @()
        if ($null -ne $hfProfile.output_token_ids) {
            $hfTokens = @($hfProfile.output_token_ids | ForEach-Object { [int]$_ })
        }

        $promptComparison = Compare-IntSequences -Expected $qwenPromptTokens -Actual $hfPromptTokens
        $tokenComparison = Compare-IntSequences -Expected $qwenTokens -Actual $hfTokens
        $promptParityPass = [bool]$promptComparison.is_match
        $tokenParityPass = [bool]$tokenComparison.is_match
        $parityPass = $promptParityPass -and $tokenParityPass

        if ($parityPass) {
            Write-Host ("  parity=PASS generated_tokens={0}" -f $qwenTokens.Count) -ForegroundColor Green
        } else {
            Write-Host (
                "  parity=FAIL prompt_match={0} token_match={1} token_first_mismatch={2} qwen={3} hf={4}" -f `
                    $promptParityPass,
                    $tokenParityPass,
                    $tokenComparison.first_mismatch_index,
                    $tokenComparison.expected_mismatch_token,
                    $tokenComparison.actual_mismatch_token
            ) -ForegroundColor Red
            $failures.Add($prompt.name)
        }

        $row = [PSCustomObject]@{
            timestamp_utc = [DateTime]::UtcNow.ToString("o")
            run_label = $RunLabel
            prompt_index = $promptIndex
            prompt_name = $prompt.name
            prompt_mode = $prompt.mode
            prompt_text = $prompt.text
            hf_device = $HFDevice
            hf_dtype = $HFDtype
            hf_model_auto_class = $HFModelAutoClass
            max_new_tokens = $MaxNewTokens
            max_context = $MaxContext
            temperature = To-InvariantString $Temperature
            top_p = To-InvariantString $TopP
            top_k = $TopK
            repeat_penalty = To-InvariantString $RepeatPenalty
            seed = $Seed
            prompt_token_parity_pass = if ($promptParityPass) { "true" } else { "false" }
            token_parity_pass = if ($tokenParityPass) { "true" } else { "false" }
            prompt_first_mismatch_index = $promptComparison.first_mismatch_index
            prompt_qwen_mismatch_token = $promptComparison.expected_mismatch_token
            prompt_hf_mismatch_token = $promptComparison.actual_mismatch_token
            first_mismatch_index = $tokenComparison.first_mismatch_index
            qwen_mismatch_token = $tokenComparison.expected_mismatch_token
            hf_mismatch_token = $tokenComparison.actual_mismatch_token
            qwen_generated_tokens = $qwenTokens.Count
            hf_generated_tokens = $hfTokens.Count
            qwen_tokens_per_second = To-InvariantString $qwenProfile.tokens_per_second
            hf_tokens_per_second = To-InvariantString $hfProfile.tokens_per_second
            qwen_profile_json = $qwenProfilePath
            hf_profile_json = $hfProfilePath
        }

        if (Test-Path -LiteralPath $resolvedCsvOut) {
            $row | Export-Csv -LiteralPath $resolvedCsvOut -NoTypeInformation -Append
        } else {
            $row | Export-Csv -LiteralPath $resolvedCsvOut -NoTypeInformation
        }
    } finally {
        if (-not $KeepProfiles.IsPresent) {
            if (Test-Path -LiteralPath $qwenProfilePath) {
                Remove-Item -LiteralPath $qwenProfilePath -Force
            }
            if (Test-Path -LiteralPath $hfProfilePath) {
                Remove-Item -LiteralPath $hfProfilePath -Force
            }
        }
    }
}

$passCount = $prompts.Count - $failures.Count
Write-Host ("Transformers parity complete: pass={0} fail={1}" -f $passCount, $failures.Count) -ForegroundColor Green
Write-Host ("CSV written to: {0}" -f $resolvedCsvOut) -ForegroundColor Green

if ($failures.Count -gt 0) {
    Write-Host ("Failed prompts: {0}" -f ($failures -join ", ")) -ForegroundColor Yellow
    if (-not $AllowMismatch.IsPresent) {
        throw "Transformers parity mismatches detected."
    }
}
