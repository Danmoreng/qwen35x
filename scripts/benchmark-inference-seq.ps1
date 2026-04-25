[CmdletBinding()]
param(
    [string]$Executable = "build/qwen35x.exe",
    [string]$HFModelDir = "models/qwen3.5-0.8b",
    [string]$CsvOut = "benchmarks/qwen35x-inference-seq.csv",
    [string]$RunLabel = "",
    [ValidateSet("gpu-bf16", "gpu-f32", "cpu-reference")]
    [string[]]$Modes = @("gpu-bf16", "gpu-f32"),
    [ValidateSet("chat-user", "prompt-text", "prompt-file", "prompt-tokens")]
    [string]$PromptMode = "chat-user",
    [string]$PromptName = "chat_short_joke",
    [string]$PromptText = "Tell me a short joke.",
    [string]$PromptFile = "",
    [string]$PromptTokensCsv = "",
    [int]$Runs = 3,
    [int]$WarmupRuns = 1,
    [int]$MaxNewTokens = 128,
    [int]$MaxContext = 256,
    [double]$Temperature = 0.0,
    [double]$TopP = 0.8,
    [int]$TopK = 20,
    [double]$RepeatPenalty = 1.05,
    [int64]$Seed = 123,
    [ValidateSet("default", "replay", "batched")]
    [string]$LucePrefillMode = "default",
    [switch]$PrefillOnly,
    [switch]$ProfileSync,
    [switch]$LuceProfile,
    [switch]$KeepProfiles,
    [string]$ProfileDir = ""
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

function To-OptionalInvariantString {
    param($Value)
    if ($null -eq $Value) {
        return ""
    }
    return [System.Convert]::ToString($Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Get-JsonProperty {
    param(
        $Object,
        [Parameter(Mandatory = $true)][string]$Name
    )

    if ($null -eq $Object) {
        return $null
    }

    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }

    return $property.Value
}

function Measure-LuceLayerMs {
    param(
        $RuntimeProfile,
        [string]$LayerType,
        [Parameter(Mandatory = $true)][string[]]$Fields
    )

    $prefill = Get-JsonProperty -Object $RuntimeProfile -Name "prefill"
    $layers = Get-JsonProperty -Object $prefill -Name "layers"
    if ($null -eq $layers) {
        return ""
    }

    $sum = 0.0
    foreach ($layer in @($layers)) {
        $currentLayerType = [string](Get-JsonProperty -Object $layer -Name "layer_type")
        if (-not [string]::IsNullOrEmpty($LayerType) -and $currentLayerType -ne $LayerType) {
            continue
        }

        foreach ($field in $Fields) {
            $value = Get-JsonProperty -Object $layer -Name $field
            if ($null -ne $value) {
                $sum += [double]$value
            }
        }
    }

    return To-InvariantString $sum
}

function Invoke-BenchmarkRun {
    param(
        [Parameter(Mandatory = $true)][string]$ExePath,
        [Parameter(Mandatory = $true)][string]$Mode,
        [Parameter(Mandatory = $true)][string]$ModelDir,
        [Parameter(Mandatory = $true)][string]$PromptMode,
        [Parameter(Mandatory = $true)][string]$PromptText,
        [Parameter(Mandatory = $false)][string]$PromptFile,
        [Parameter(Mandatory = $false)][string]$PromptTokensCsv,
        [Parameter(Mandatory = $true)][int]$MaxNewTokens,
        [Parameter(Mandatory = $true)][int]$MaxContext,
        [Parameter(Mandatory = $true)][double]$Temperature,
        [Parameter(Mandatory = $true)][double]$TopP,
        [Parameter(Mandatory = $true)][int]$TopK,
        [Parameter(Mandatory = $true)][double]$RepeatPenalty,
        [Parameter(Mandatory = $true)][int64]$Seed,
        [Parameter(Mandatory = $true)][string]$LucePrefillMode,
        [Parameter(Mandatory = $true)][bool]$PrefillOnlyEnabled,
        [Parameter(Mandatory = $true)][bool]$ProfileSyncEnabled,
        [Parameter(Mandatory = $true)][bool]$LuceProfileEnabled,
        [Parameter(Mandatory = $true)][string]$ProfileJsonPath
    )

    $args = @()
    switch ($Mode) {
        "gpu-bf16" {
            $args += @("--infer-gpu", "--gpu-bf16")
        }
        "gpu-f32" {
            $args += @("--infer-gpu", "--gpu-f32-matvec")
        }
        "cpu-reference" {
            $args += @("--infer-reference")
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
    } elseif ($PromptMode -eq "prompt-file") {
        $args += @("--prompt-file", $PromptFile)
    } else {
        $args += @("--prompt-tokens", $PromptTokensCsv)
    }

    if ($ProfileSyncEnabled -and $Mode -ne "cpu-reference") {
        $args += @("--profile-sync")
    }
    if ($LuceProfileEnabled -and $Mode -ne "cpu-reference") {
        $args += @("--luce-profile")
    }
    if ($PrefillOnlyEnabled) {
        $args += @("--prefill-only")
    }
    if ($Mode -ne "cpu-reference" -and $LucePrefillMode -ne "default") {
        $args += @("--luce-prefill-mode", $LucePrefillMode)
    }

    Write-Host ("Running mode={0} prompt={1}" -f $Mode, $PromptMode) -ForegroundColor Cyan
    $runOutput = & $ExePath @args 2>&1
    foreach ($line in $runOutput) {
        Write-Host $line
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark run failed (mode=$Mode, exit_code=$LASTEXITCODE)."
    }

    if (-not (Test-Path $ProfileJsonPath)) {
        throw "Missing profile JSON output: $ProfileJsonPath"
    }

    return Get-Content -Raw -LiteralPath $ProfileJsonPath | ConvertFrom-Json
}

$scriptDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $scriptDir

$resolvedExe = Resolve-RepoPath -Path $Executable -RepoRoot $repoRoot
$resolvedModelDir = Resolve-RepoPath -Path $HFModelDir -RepoRoot $repoRoot
$resolvedCsvOut = Resolve-RepoPath -Path $CsvOut -RepoRoot $repoRoot
$profileTmpDir = if ([string]::IsNullOrWhiteSpace($ProfileDir)) {
    Join-Path $repoRoot "build\bench-profiles"
} else {
    Resolve-RepoPath -Path $ProfileDir -RepoRoot $repoRoot
}
$resolvedPromptFile = ""
if ($PromptMode -eq "prompt-file") {
    $resolvedPromptFile = Resolve-RepoPath -Path $PromptFile -RepoRoot $repoRoot
}

if (-not (Test-Path $resolvedExe)) {
    throw "Executable not found: $resolvedExe"
}
if (-not (Test-Path $resolvedModelDir)) {
    throw "Model directory not found: $resolvedModelDir"
}
if ($Runs -lt 1) {
    throw "Runs must be >= 1."
}
if ($WarmupRuns -lt 0) {
    throw "WarmupRuns must be >= 0."
}
if (($PromptMode -eq "chat-user" -or $PromptMode -eq "prompt-text") -and [string]::IsNullOrWhiteSpace($PromptText)) {
    throw "PromptText must be non-empty for prompt mode '$PromptMode'."
}
if ($PromptMode -eq "prompt-file" -and [string]::IsNullOrWhiteSpace($PromptFile)) {
    throw "PromptFile must be non-empty when PromptMode is 'prompt-file'."
}
if ($PromptMode -eq "prompt-file" -and -not (Test-Path -LiteralPath $resolvedPromptFile)) {
    throw "PromptFile not found: $resolvedPromptFile"
}
if ($PromptMode -eq "prompt-tokens" -and [string]::IsNullOrWhiteSpace($PromptTokensCsv)) {
    throw "PromptTokensCsv must be non-empty when PromptMode is 'prompt-tokens'."
}
if ($LuceProfile.IsPresent -and $Modes -contains "cpu-reference") {
    Write-Warning "LuceProfile is ignored for cpu-reference mode."
}

New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedCsvOut) -Force | Out-Null
New-Item -ItemType Directory -Path $profileTmpDir -Force | Out-Null

Write-Host ("Sequential benchmark start: modes={0}, warmup={1}, runs={2}" -f ($Modes -join ","), $WarmupRuns, $Runs) -ForegroundColor Green
Write-Host ("CSV output: {0}" -f $resolvedCsvOut) -ForegroundColor Green
if ($KeepProfiles.IsPresent -or $LuceProfile.IsPresent) {
    Write-Host ("Profile JSON output: {0}" -f $profileTmpDir) -ForegroundColor Green
}

foreach ($mode in $Modes) {
    for ($warm = 1; $warm -le $WarmupRuns; ++$warm) {
        $warmProfile = Join-Path $profileTmpDir ("warmup_{0}_{1}_{2}.json" -f $mode, $warm, [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())
        try {
            $null = Invoke-BenchmarkRun `
                -ExePath $resolvedExe `
                -Mode $mode `
                -ModelDir $resolvedModelDir `
                -PromptMode $PromptMode `
                -PromptText $PromptText `
                -PromptFile $resolvedPromptFile `
                -PromptTokensCsv $PromptTokensCsv `
                -MaxNewTokens $MaxNewTokens `
                -MaxContext $MaxContext `
                -Temperature $Temperature `
                -TopP $TopP `
                -TopK $TopK `
                -RepeatPenalty $RepeatPenalty `
                -Seed $Seed `
                -LucePrefillMode $LucePrefillMode `
                -PrefillOnlyEnabled $PrefillOnly.IsPresent `
                -ProfileSyncEnabled $ProfileSync.IsPresent `
                -LuceProfileEnabled $LuceProfile.IsPresent `
                -ProfileJsonPath $warmProfile
            Write-Host ("Warmup completed: mode={0} run={1}/{2}" -f $mode, $warm, $WarmupRuns) -ForegroundColor DarkGray
        } finally {
            if ((Test-Path $warmProfile) -and -not $KeepProfiles.IsPresent) {
                Remove-Item -LiteralPath $warmProfile -Force
            }
        }
    }

    for ($runIndex = 1; $runIndex -le $Runs; ++$runIndex) {
        $profilePath = Join-Path $profileTmpDir ("run_{0}_{1}_{2}.json" -f $mode, $runIndex, [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())
        try {
            $profile = Invoke-BenchmarkRun `
                -ExePath $resolvedExe `
                -Mode $mode `
                -ModelDir $resolvedModelDir `
                -PromptMode $PromptMode `
                -PromptText $PromptText `
                -PromptFile $resolvedPromptFile `
                -PromptTokensCsv $PromptTokensCsv `
                -MaxNewTokens $MaxNewTokens `
                -MaxContext $MaxContext `
                -Temperature $Temperature `
                -TopP $TopP `
                -TopK $TopK `
                -RepeatPenalty $RepeatPenalty `
                -Seed $Seed `
                -LucePrefillMode $LucePrefillMode `
                -PrefillOnlyEnabled $PrefillOnly.IsPresent `
                -ProfileSyncEnabled $ProfileSync.IsPresent `
                -LuceProfileEnabled $LuceProfile.IsPresent `
                -ProfileJsonPath $profilePath

            $luceProfileJson = Get-JsonProperty -Object $profile -Name "luce_profile"
            $lucePrefillJson = Get-JsonProperty -Object $luceProfileJson -Name "prefill"
            $luceDecodeJson = Get-JsonProperty -Object $luceProfileJson -Name "decode"
            $luceProfileEnabledValue = Get-JsonProperty -Object $luceProfileJson -Name "enabled"
            $profilePathForCsv = ""
            if ($KeepProfiles.IsPresent -or $LuceProfile.IsPresent) {
                $profilePathForCsv = $profilePath
            }
            $effectiveLucePrefillMode = [string](Get-JsonProperty -Object $profile -Name "luce_prefill_mode")
            if ([string]::IsNullOrWhiteSpace($effectiveLucePrefillMode)) {
                $effectiveLucePrefillMode = $LucePrefillMode
            }

            $row = [PSCustomObject]@{
                timestamp_utc    = [DateTime]::UtcNow.ToString("o")
                run_label        = $RunLabel
                mode             = $mode
                luce_prefill_mode = $effectiveLucePrefillMode
                prefill_only     = [bool]$profile.prefill_only
                run_index        = $runIndex
                prompt_name      = $PromptName
                prompt_tokens    = [int]$profile.prompt_tokens
                generated_tokens = [int]$profile.generated_tokens
                load_time_ms     = To-InvariantString $profile.load_time_ms
                prefill_time_ms  = To-InvariantString $profile.prefill_time_ms
                prefill_tokens_per_second = To-InvariantString $profile.prefill_tokens_per_second
                decode_time_ms   = To-InvariantString $profile.decode_time_ms
                tokens_per_second = To-InvariantString $profile.tokens_per_second
                profile_json     = $profilePathForCsv
                luce_profile_enabled = [bool]$luceProfileEnabledValue
                luce_prefill_host_total_ms = To-OptionalInvariantString (Get-JsonProperty -Object $lucePrefillJson -Name "host_total_ms")
                luce_prefill_gpu_total_ms = To-OptionalInvariantString (Get-JsonProperty -Object $lucePrefillJson -Name "gpu_total_ms")
                luce_prefill_token_upload_ms = To-OptionalInvariantString (Get-JsonProperty -Object $lucePrefillJson -Name "token_upload_ms")
                luce_prefill_embed_ms = To-OptionalInvariantString (Get-JsonProperty -Object $lucePrefillJson -Name "embed_ms")
                luce_prefill_mark_seen_ms = To-OptionalInvariantString (Get-JsonProperty -Object $lucePrefillJson -Name "mark_seen_ms")
                luce_prefill_final_norm_ms = To-OptionalInvariantString (Get-JsonProperty -Object $lucePrefillJson -Name "final_norm_ms")
                luce_prefill_lm_head_ms = To-OptionalInvariantString (Get-JsonProperty -Object $lucePrefillJson -Name "lm_head_ms")
                luce_prefill_lm_reduce_ms = To-OptionalInvariantString (Get-JsonProperty -Object $lucePrefillJson -Name "lm_reduce_ms")
                luce_prefill_hidden_handoff_ms = To-OptionalInvariantString (Get-JsonProperty -Object $lucePrefillJson -Name "hidden_handoff_ms")
                luce_prefill_output_token_download_ms = To-OptionalInvariantString (Get-JsonProperty -Object $lucePrefillJson -Name "output_token_download_ms")
                luce_prefill_deltanet_total_ms = Measure-LuceLayerMs -RuntimeProfile $luceProfileJson -LayerType "deltanet" -Fields @("total_ms")
                luce_prefill_deltanet_recurrence_ms = Measure-LuceLayerMs -RuntimeProfile $luceProfileJson -LayerType "deltanet" -Fields @("recurrence_ms")
                luce_prefill_deltanet_projection_ms = Measure-LuceLayerMs -RuntimeProfile $luceProfileJson -LayerType "deltanet" -Fields @("qkv_projection_ms", "z_projection_ms", "beta_alpha_projection_ms", "out_projection_ms")
                luce_prefill_full_attention_total_ms = Measure-LuceLayerMs -RuntimeProfile $luceProfileJson -LayerType "full_attention" -Fields @("total_ms")
                luce_prefill_full_attention_attention_ms = Measure-LuceLayerMs -RuntimeProfile $luceProfileJson -LayerType "full_attention" -Fields @("attention_ms")
                luce_prefill_full_attention_qk_norm_rope_ms = Measure-LuceLayerMs -RuntimeProfile $luceProfileJson -LayerType "full_attention" -Fields @("qk_norm_rope_ms")
                luce_prefill_full_attention_projection_ms = Measure-LuceLayerMs -RuntimeProfile $luceProfileJson -LayerType "full_attention" -Fields @("qkv_projection_ms", "kv_projection_ms", "out_projection_ms")
                luce_prefill_mlp_total_ms = Measure-LuceLayerMs -RuntimeProfile $luceProfileJson -LayerType "" -Fields @("mlp_norm_ms", "mlp_projection_ms", "mlp_activation_ms", "mlp_down_projection_ms", "mlp_residual_ms")
                luce_decode_steps = To-OptionalInvariantString (Get-JsonProperty -Object $luceDecodeJson -Name "steps")
                luce_decode_host_total_ms = To-OptionalInvariantString (Get-JsonProperty -Object $luceDecodeJson -Name "host_total_ms")
                luce_decode_seen_token_upload_ms = To-OptionalInvariantString (Get-JsonProperty -Object $luceDecodeJson -Name "seen_token_upload_ms")
                luce_decode_launch_total_ms = To-OptionalInvariantString (Get-JsonProperty -Object $luceDecodeJson -Name "launch_total_ms")
                luce_decode_kernel_ms = To-OptionalInvariantString (Get-JsonProperty -Object $luceDecodeJson -Name "decode_kernel_ms")
                luce_decode_lm_head_ms = To-OptionalInvariantString (Get-JsonProperty -Object $luceDecodeJson -Name "lm_head_ms")
                luce_decode_output_token_download_ms = To-OptionalInvariantString (Get-JsonProperty -Object $luceDecodeJson -Name "output_token_download_ms")
            }

            if (Test-Path $resolvedCsvOut) {
                $row | Export-Csv -LiteralPath $resolvedCsvOut -NoTypeInformation -Append
            } else {
                $row | Export-Csv -LiteralPath $resolvedCsvOut -NoTypeInformation
            }

            Write-Host ("Recorded: mode={0} run={1}/{2} tps={3}" -f $mode, $runIndex, $Runs, $row.tokens_per_second) -ForegroundColor Yellow
            if ($profilePathForCsv -ne "") {
                Write-Host ("Profile JSON kept: {0}" -f $profilePathForCsv) -ForegroundColor DarkGray
            }
        } finally {
            if ((Test-Path $profilePath) -and -not $KeepProfiles.IsPresent -and -not $LuceProfile.IsPresent) {
                Remove-Item -LiteralPath $profilePath -Force
            }
        }
    }
}

Write-Host "Benchmark complete." -ForegroundColor Green
Write-Host ("CSV written to: {0}" -f $resolvedCsvOut) -ForegroundColor Green
