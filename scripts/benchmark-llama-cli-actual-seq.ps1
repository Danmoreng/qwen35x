[CmdletBinding()]
param(
    [string]$LlamaExe = "third_party/reference/llama.cpp/build-qwen35x/bin/llama-completion.exe",
    [string]$Model = "models/gguf/qwen3.5-0.8b-bf16.gguf",
    [string]$PromptFile = "benchmarks/inputs/wiki_artificial_intelligence_64k_prompt.txt",
    [string]$OutputDir = "benchmarks/llama-cli/wiki-ai-64k-gen128",
    [string]$CsvOut = "benchmarks/llama-cli/qwen35x-wiki-ai-64k-gen128.csv",
    [string]$RunLabel = "llama-cli-wiki-ai-64k-gen128",
    [ValidateSet("off", "on", "both")]
    [string]$FlashAttention = "both",
    [int]$MaxContext = 65536,
    [int]$MaxNewTokens = 128,
    [int]$GpuLayers = 99,
    [int]$BatchSize = 2048,
    [int]$UBatchSize = 512,
    [double]$Temperature = 0.0,
    [double]$TopP = 0.8,
    [int]$TopK = 20,
    [double]$RepeatPenalty = 1.05,
    [int]$Seed = 123,
    [int]$TimeoutSeconds = 1800,
    [switch]$Warmup
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

function Get-TimingMetric {
    param(
        [Parameter(Mandatory = $true)][string]$Text,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $escaped = [regex]::Escape($Label)
    $pattern = "(?m)^(?:[^\r\n:]+:\s*)?\s*${escaped}\s*=\s*(?<ms>[0-9]+(?:\.[0-9]+)?)\s*ms(?:\s*/\s*(?<tokens>[0-9]+)\s*(?:tokens|runs))?(?:.*?\(\s*(?<per_token>[0-9]+(?:\.[0-9]+)?)\s*ms per token,\s*(?<tps>[0-9]+(?:\.[0-9]+)?|inf)\s*tokens per second\))?"
    $match = [regex]::Match($Text, $pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if (-not $match.Success) {
        return [PSCustomObject]@{
            ms = ""
            tokens = ""
            tokens_per_second = ""
        }
    }

    return [PSCustomObject]@{
        ms = $match.Groups["ms"].Value
        tokens = $match.Groups["tokens"].Value
        tokens_per_second = $match.Groups["tps"].Value
    }
}

function Invoke-LlamaCompletionRun {
    param(
        [Parameter(Mandatory = $true)][string]$ExePath,
        [Parameter(Mandatory = $true)][string]$ModelPath,
        [Parameter(Mandatory = $true)][string]$PromptPath,
        [Parameter(Mandatory = $true)][string]$ResolvedOutputDir,
        [Parameter(Mandatory = $true)][string]$FlashMode
    )

    $safeMode = if ($FlashMode -eq "on") { "fa" } else { "no-fa" }
    $stdoutPath = Join-Path $ResolvedOutputDir ("llama-completion_{0}.stdout.txt" -f $safeMode)
    $stderrPath = Join-Path $ResolvedOutputDir ("llama-completion_{0}.stderr.txt" -f $safeMode)
    $generatedPath = Join-Path $ResolvedOutputDir ("llama-completion_{0}.generated.txt" -f $safeMode)
    $commandPath = Join-Path $ResolvedOutputDir ("llama-completion_{0}.command.txt" -f $safeMode)

    Remove-Item -LiteralPath $stdoutPath, $stderrPath, $generatedPath, $commandPath -Force -ErrorAction SilentlyContinue

    $args = @(
        "-m", $ModelPath,
        "-f", $PromptPath,
        "-n", "$MaxNewTokens",
        "-c", "$MaxContext",
        "-ngl", "$GpuLayers",
        "-b", "$BatchSize",
        "-ub", "$UBatchSize",
        "--flash-attn", $FlashMode,
        "--temp", (To-InvariantString $Temperature),
        "--top-p", (To-InvariantString $TopP),
        "--top-k", "$TopK",
        "--repeat-penalty", (To-InvariantString $RepeatPenalty),
        "--seed", "$Seed",
        "--no-display-prompt",
        "-no-cnv",
        "--perf",
        "--simple-io"
    )
    if (-not $Warmup.IsPresent) {
        $args += "--no-warmup"
    }

    $quoted = @($ExePath) + ($args | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    })
    Set-Content -LiteralPath $commandPath -Value ($quoted -join " ") -Encoding UTF8

    $start = Get-Date
    Write-Host ("Starting llama-completion flash_attn={0}; stdout/stderr redirected to {1}" -f $FlashMode, $ResolvedOutputDir) -ForegroundColor Cyan
    $process = Start-Process `
        -FilePath $ExePath `
        -ArgumentList $args `
        -WorkingDirectory (Split-Path -Parent $ExePath) `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru

    $completed = $process.WaitForExit($TimeoutSeconds * 1000)
    if (-not $completed) {
        try {
            $process.Kill($true)
        } catch {
            $process.Kill()
        }
        throw "llama-completion timed out after $TimeoutSeconds seconds for flash_attn=$FlashMode."
    }

    $end = Get-Date
    $wallMs = ($end - $start).TotalMilliseconds
    if ($process.ExitCode -ne 0) {
        throw "llama-completion failed for flash_attn=$FlashMode with exit code $($process.ExitCode). See $stderrPath"
    }

    $stdoutText = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -Raw -LiteralPath $stdoutPath } else { "" }
    $stderrText = if (Test-Path -LiteralPath $stderrPath) { Get-Content -Raw -LiteralPath $stderrPath } else { "" }
    $combined = $stdoutText + "`n" + $stderrText
    $generated = $stdoutText.TrimEnd()
    Set-Content -LiteralPath $generatedPath -Value $generated -Encoding UTF8

    $load = Get-TimingMetric -Text $combined -Label "load time"
    $promptEval = Get-TimingMetric -Text $combined -Label "prompt eval time"
    $eval = Get-TimingMetric -Text $combined -Label "eval time"
    $total = Get-TimingMetric -Text $combined -Label "total time"

    return [PSCustomObject]@{
        timestamp_utc = [DateTime]::UtcNow.ToString("o")
        run_label = $RunLabel
        backend = "llama-completion"
        flash_attn = $FlashMode
        model = $ModelPath
        prompt_file = $PromptPath
        max_context = $MaxContext
        max_new_tokens = $MaxNewTokens
        batch_size = $BatchSize
        ubatch_size = $UBatchSize
        gpu_layers = $GpuLayers
        temperature = To-InvariantString $Temperature
        top_p = To-InvariantString $TopP
        top_k = $TopK
        repeat_penalty = To-InvariantString $RepeatPenalty
        seed = $Seed
        wall_time_ms = To-InvariantString $wallMs
        load_time_ms = $load.ms
        prompt_eval_tokens = $promptEval.tokens
        prompt_eval_time_ms = $promptEval.ms
        prompt_eval_tokens_per_second = $promptEval.tokens_per_second
        eval_tokens = $eval.tokens
        eval_time_ms = $eval.ms
        eval_tokens_per_second = $eval.tokens_per_second
        total_time_ms = $total.ms
        generated_chars = $generated.Length
        stdout_path = $stdoutPath
        stderr_path = $stderrPath
        generated_text_path = $generatedPath
        command_path = $commandPath
    }
}

$scriptDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $scriptDir

$resolvedExe = Resolve-RepoPath -Path $LlamaExe -RepoRoot $repoRoot
$resolvedModel = Resolve-RepoPath -Path $Model -RepoRoot $repoRoot
$resolvedPrompt = Resolve-RepoPath -Path $PromptFile -RepoRoot $repoRoot
$resolvedOutputDir = Resolve-RepoPath -Path $OutputDir -RepoRoot $repoRoot
$resolvedCsvOut = Resolve-RepoPath -Path $CsvOut -RepoRoot $repoRoot

if (-not (Test-Path -LiteralPath $resolvedExe)) {
    throw "llama-completion executable not found: $resolvedExe"
}
if (-not (Test-Path -LiteralPath $resolvedModel)) {
    throw "GGUF model not found: $resolvedModel"
}
if (-not (Test-Path -LiteralPath $resolvedPrompt)) {
    throw "Prompt file not found: $resolvedPrompt"
}
if ($MaxContext -lt 1) {
    throw "MaxContext must be >= 1."
}
if ($MaxNewTokens -lt 1) {
    throw "MaxNewTokens must be >= 1."
}
if ($TimeoutSeconds -lt 1) {
    throw "TimeoutSeconds must be >= 1."
}

New-Item -ItemType Directory -Path $resolvedOutputDir -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedCsvOut) -Force | Out-Null

$modes = if ($FlashAttention -eq "both") { @("off", "on") } else { @($FlashAttention) }
$rows = @()
foreach ($mode in $modes) {
    $row = Invoke-LlamaCompletionRun `
        -ExePath $resolvedExe `
        -ModelPath $resolvedModel `
        -PromptPath $resolvedPrompt `
        -ResolvedOutputDir $resolvedOutputDir `
        -FlashMode $mode
    $rows += $row

    if (Test-Path -LiteralPath $resolvedCsvOut) {
        $row | Export-Csv -LiteralPath $resolvedCsvOut -NoTypeInformation -Append
    } else {
        $row | Export-Csv -LiteralPath $resolvedCsvOut -NoTypeInformation
    }

    Write-Host ("Completed flash_attn={0}: prompt_tps={1}, gen_tps={2}, generated={3}" -f `
            $mode, $row.prompt_eval_tokens_per_second, $row.eval_tokens_per_second, $row.generated_text_path) -ForegroundColor Green
}

Write-Host ("CSV written to: {0}" -f $resolvedCsvOut) -ForegroundColor Green
Write-Host ("Artifacts written to: {0}" -f $resolvedOutputDir) -ForegroundColor Green
