[CmdletBinding()]
param(
    [int[]]$Contexts = @(256, 512, 1024, 2048, 4096),
    [int]$MaxNewTokens = 128,
    [int]$Runs = 3,
    [int]$WarmupRuns = 1,
    [int]$LlamaRepetitions = 3,
    [string]$OutDir = "benchmarks/model-matrix",
    [string]$SummaryCsvOut = "benchmarks/model-matrix/qwen35x-vs-llama-matrix-summary.csv",
    [string]$TokenId = "198",
    [string]$PromptSourceFile = "benchmarks/inputs/wiki_artificial_intelligence_64k_prompt.txt",
    [int]$PromptSourceTokens = 65405,
    [switch]$BuildQwenVariants,
    [switch]$BuildLlama,
    [switch]$CleanLlamaBuild,
    [string]$LlamaExe = "third_party/reference/llama.cpp/build-qwen35x/bin/llama-completion.exe",
    [string]$Qwen08Exe = "build-0p8b-bench/qwen35x.exe",
    [string]$Qwen4BExe = "build-4b-bench/qwen35x.exe",
    [string]$Qwen08ModelDir = "models/qwen3.5-0.8b",
    [string]$Qwen4BModelDir = "models/qwen3.5-4b",
    [string]$Llama08Gguf = "models/gguf/qwen3.5-0.8b-bf16.gguf",
    [string]$Llama4BGguf = "models/gguf/qwen3.5-4b-bf16.gguf",
    [int]$LlamaGpuLayers = 99,
    [int]$LlamaBatchSize = 2048,
    [int]$LlamaUBatchSize = 512,
    [int]$LlamaTimeoutSeconds = 1800,
    [switch]$LlamaNoWarmup,
    [switch]$SkipQwen,
    [switch]$SkipLlama,
    [switch]$SummarizeOnly
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
    param($Value)
    if ($null -eq $Value -or $Value -eq "") {
        return ""
    }
    return [System.Convert]::ToString($Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function New-PromptTokenCsv {
    param(
        [Parameter(Mandatory = $true)][int]$Count,
        [Parameter(Mandatory = $true)][string]$Token
    )
    if ($Count -lt 1) {
        throw "Prompt token count must be >= 1."
    }
    $tokens = for ($i = 0; $i -lt $Count; ++$i) {
        $Token
    }
    return $tokens -join ","
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][scriptblock]$Script
    )
    Write-Host $Label -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    & $Script
    $exitCode = Get-Variable -Name LASTEXITCODE -Scope Global -ErrorAction SilentlyContinue
    if ($null -ne $exitCode -and $null -ne $exitCode.Value -and $exitCode.Value -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

function Import-QwenRows {
    param(
        [Parameter(Mandatory = $true)][string]$CsvPath,
        [Parameter(Mandatory = $true)][string]$Model,
        [Parameter(Mandatory = $true)][int]$ContextTokens,
        [Parameter(Mandatory = $true)][string]$Workload
    )
    $csvRows = @(Import-Csv -LiteralPath $CsvPath)
    if ($csvRows.Count -eq 0) {
        return @()
    }
    $first = $csvRows[0]
    $prefillTps = ($csvRows | Measure-Object -Property prefill_tokens_per_second -Average).Average
    $genTps = ($csvRows | Measure-Object -Property tokens_per_second -Average).Average
    $loadMs = ($csvRows | Measure-Object -Property load_time_ms -Average).Average
    $prefillMs = ($csvRows | Measure-Object -Property prefill_time_ms -Average).Average
    $decodeMs = ($csvRows | Measure-Object -Property decode_time_ms -Average).Average

    return @([PSCustomObject]@{
        timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
        model = $Model
        implementation = "qwen35x"
        backend = $first.mode
        flash_attention = ""
        workload = $Workload
        context_tokens = $ContextTokens
        prompt_tokens = $first.prompt_tokens
        generated_tokens = $first.generated_tokens
        load_time_ms = To-InvariantString $loadMs
        prefill_time_ms = To-InvariantString $prefillMs
        prefill_tokens_per_second = To-InvariantString $prefillTps
        decode_time_ms = To-InvariantString $decodeMs
        generation_tokens_per_second = if ($Workload -eq "generation") { To-InvariantString $genTps } else { "" }
        source_file = $CsvPath
    })
}

function Import-LlamaRows {
    param(
        [Parameter(Mandatory = $true)][string]$JsonPath,
        [Parameter(Mandatory = $true)][string]$Model,
        [Parameter(Mandatory = $true)][int]$ContextTokens,
        [Parameter(Mandatory = $true)][string]$Workload,
        [Parameter(Mandatory = $true)][string]$FlashAttention
    )
    $json = @(Get-Content -Raw -LiteralPath $JsonPath | ConvertFrom-Json)
    $rows = @()
    foreach ($row in $json) {
        $promptTokens = [int]$row.n_prompt
        $genTokens = [int]$row.n_gen
        if ($Workload -eq "prefill" -and ($promptTokens -le 0 -or $genTokens -ne 0)) {
            continue
        }
        if ($Workload -eq "generation" -and $genTokens -le 0) {
            continue
        }
        $avgTs = $row.avg_ts
        $backend = if ($row.PSObject.Properties.Name -contains "backend") {
            [string]$row.backend
        } elseif ($row.PSObject.Properties.Name -contains "backends") {
            [string]$row.backends
        } else {
            ""
        }
        $rows += [PSCustomObject]@{
            timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
            model = $Model
            implementation = "llama.cpp"
            backend = $backend
            flash_attention = $FlashAttention
            workload = $Workload
            context_tokens = $ContextTokens
            prompt_tokens = $promptTokens
            generated_tokens = $genTokens
            load_time_ms = ""
            prefill_time_ms = ""
            prefill_tokens_per_second = if ($Workload -eq "prefill") { To-InvariantString $avgTs } else { "" }
            decode_time_ms = ""
            generation_tokens_per_second = if ($Workload -eq "generation") { To-InvariantString $avgTs } else { "" }
            source_file = $JsonPath
        }
    }
    return $rows
}

function New-ImplementationName {
    param(
        [Parameter(Mandatory = $true)][object]$Row
    )
    if ($Row.implementation -eq "llama.cpp") {
        if ($Row.flash_attention -eq "on") {
            return "llama.cpp-fa"
        }
        return "llama.cpp-no-fa"
    }
    return [string]$Row.implementation
}

function Format-Throughput {
    param($Value)
    if ($null -eq $Value -or $Value -eq "") {
        return ""
    }
    $parsed = 0.0
    if (-not [double]::TryParse(
            [string]$Value,
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$parsed)) {
        return [string]$Value
    }
    return $parsed.ToString("F2", [System.Globalization.CultureInfo]::InvariantCulture)
}

function New-CompactComparisonRows {
    param(
        [Parameter(Mandatory = $true)][object[]]$Rows
    )
    $compactRows = @()
    $groups = $Rows | Group-Object model, context_tokens, implementation, flash_attention
    foreach ($group in $groups) {
        $first = $group.Group[0]
        $prefillRow = @($group.Group | Where-Object { $_.workload -eq "prefill" } | Select-Object -First 1)
        $generationRow = @($group.Group | Where-Object { $_.workload -eq "generation" } | Select-Object -First 1)

        $compactRows += [PSCustomObject]@{
            model = $first.model
            implementation = New-ImplementationName -Row $first
            ctx = [int]$first.context_tokens
            prefill_tokens_per_second = if ($prefillRow.Count -gt 0) { Format-Throughput $prefillRow[0].prefill_tokens_per_second } else { "" }
            generation_tokens_per_second = if ($generationRow.Count -gt 0) { Format-Throughput $generationRow[0].generation_tokens_per_second } else { "" }
        }
    }

    return @($compactRows | Sort-Object ctx, model, implementation)
}

function Export-CompactComparisonCsvs {
    param(
        [Parameter(Mandatory = $true)][object[]]$Rows,
        [Parameter(Mandatory = $true)][string]$OutDir,
        [Parameter(Mandatory = $true)][string]$SummaryCsvOut
    )
    $compactRows = @(New-CompactComparisonRows -Rows $Rows)
    if ($compactRows.Count -eq 0) {
        throw "No compact comparison rows were produced."
    }

    $compactRows | Export-Csv -LiteralPath $SummaryCsvOut -NoTypeInformation
    foreach ($contextGroup in ($compactRows | Group-Object ctx)) {
        $ctx = [int]$contextGroup.Name
        $ctxCsv = Join-Path $OutDir ("comparison-ctx{0}.csv" -f $ctx)
        $contextGroup.Group |
            Sort-Object model, implementation |
            Export-Csv -LiteralPath $ctxCsv -NoTypeInformation
    }

    return $compactRows
}

function New-PromptFileForContext {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][int]$SourceTokens,
        [Parameter(Mandatory = $true)][int]$ContextTokens,
        [Parameter(Mandatory = $true)][string]$OutDir
    )
    if ($SourceTokens -lt 1) {
        throw "PromptSourceTokens must be >= 1."
    }
    if (-not (Test-Path -LiteralPath $SourcePath)) {
        throw "Prompt source file not found: $SourcePath"
    }

    $sourceText = Get-Content -Raw -LiteralPath $SourcePath
    if ([string]::IsNullOrWhiteSpace($sourceText)) {
        throw "Prompt source file is empty: $SourcePath"
    }

    $charCount = [Math]::Max(1, [Math]::Min($sourceText.Length, [int][Math]::Ceiling($sourceText.Length * ($ContextTokens / [double]$SourceTokens))))
    $promptText = $sourceText.Substring(0, $charCount)
    $promptPath = Join-Path $OutDir ("prompt-ctx{0}.txt" -f $ContextTokens)
    Set-Content -LiteralPath $promptPath -Value $promptText -Encoding UTF8
    return $promptPath
}

function Import-LlamaActualRows {
    param(
        [Parameter(Mandatory = $true)][string]$CsvPath,
        [Parameter(Mandatory = $true)][string]$Model,
        [Parameter(Mandatory = $true)][int]$ContextTokens
    )
    $csvRows = @(Import-Csv -LiteralPath $CsvPath)
    $rows = @()
    foreach ($row in $csvRows) {
        $rows += [PSCustomObject]@{
            timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
            model = $Model
            implementation = "llama.cpp"
            backend = $row.backend
            flash_attention = $row.flash_attn
            workload = "prefill"
            context_tokens = $ContextTokens
            prompt_tokens = $row.prompt_eval_tokens
            generated_tokens = ""
            load_time_ms = $row.load_time_ms
            prefill_time_ms = $row.prompt_eval_time_ms
            prefill_tokens_per_second = $row.prompt_eval_tokens_per_second
            decode_time_ms = ""
            generation_tokens_per_second = ""
            source_file = $CsvPath
        }
        $rows += [PSCustomObject]@{
            timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
            model = $Model
            implementation = "llama.cpp"
            backend = $row.backend
            flash_attention = $row.flash_attn
            workload = "generation"
            context_tokens = $ContextTokens
            prompt_tokens = $row.prompt_eval_tokens
            generated_tokens = $row.eval_tokens
            load_time_ms = $row.load_time_ms
            prefill_time_ms = ""
            prefill_tokens_per_second = ""
            decode_time_ms = $row.eval_time_ms
            generation_tokens_per_second = $row.eval_tokens_per_second
            source_file = $CsvPath
        }
    }
    return $rows
}

function Import-ExistingQwenRows {
    param(
        [Parameter(Mandatory = $true)][object[]]$Models,
        [Parameter(Mandatory = $true)][int[]]$Contexts,
        [Parameter(Mandatory = $true)][string]$OutDir,
        [Parameter(Mandatory = $true)][int]$MaxNewTokens
    )
    $rows = @()
    foreach ($model in $Models) {
        $safeModel = $model.name -replace '[^A-Za-z0-9]+', ''
        foreach ($context in $Contexts) {
            $qwenPrefillCsv = Join-Path $OutDir ("qwen35x-{0}-ctx{1}-prefill.csv" -f $safeModel, $context)
            if (Test-Path -LiteralPath $qwenPrefillCsv) {
                $rows += Import-QwenRows -CsvPath $qwenPrefillCsv -Model $model.name -ContextTokens $context -Workload "prefill"
            }

            $qwenGenCsv = Join-Path $OutDir ("qwen35x-{0}-ctx{1}-gen{2}.csv" -f $safeModel, $context, $MaxNewTokens)
            if (Test-Path -LiteralPath $qwenGenCsv) {
                $rows += Import-QwenRows -CsvPath $qwenGenCsv -Model $model.name -ContextTokens $context -Workload "generation"
            }
        }
    }
    return $rows
}

function Import-ExistingLlamaRows {
    param(
        [Parameter(Mandatory = $true)][object[]]$Models,
        [Parameter(Mandatory = $true)][int[]]$Contexts,
        [Parameter(Mandatory = $true)][string]$OutDir,
        [Parameter(Mandatory = $true)][int]$MaxNewTokens
    )
    $rows = @()
    foreach ($model in $Models) {
        $safeModel = $model.name -replace '[^A-Za-z0-9]+', ''
        foreach ($context in $Contexts) {
            foreach ($flash in @("off", "on")) {
                $flashSuffix = if ($flash -eq "on") { "fa" } else { "no-fa" }
                $llamaActualCsv = Join-Path $OutDir ("llama-{0}-{1}-ctx{2}-actual-gen{3}.csv" -f $safeModel, $flashSuffix, $context, $MaxNewTokens)
                if (Test-Path -LiteralPath $llamaActualCsv) {
                    $rows += Import-LlamaActualRows -CsvPath $llamaActualCsv -Model $model.name -ContextTokens $context
                }
            }
        }
    }
    return $rows
}

$scriptDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $scriptDir

$resolvedOutDir = Resolve-RepoPath -Path $OutDir -RepoRoot $repoRoot
$resolvedSummaryCsvOut = Resolve-RepoPath -Path $SummaryCsvOut -RepoRoot $repoRoot
New-Item -ItemType Directory -Path $resolvedOutDir -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedSummaryCsvOut) -Force | Out-Null

$qwenBench = Join-Path $scriptDir "benchmark-inference-seq.ps1"
$llamaBench = Join-Path $scriptDir "benchmark-llama-bf16.ps1"
$llamaActualBench = Join-Path $scriptDir "benchmark-llama-cli-actual-seq.ps1"
$buildScript = Join-Path $scriptDir "build.ps1"
$resolvedPromptSourceFile = Resolve-RepoPath -Path $PromptSourceFile -RepoRoot $repoRoot
$resolvedLlamaExe = Resolve-RepoPath -Path $LlamaExe -RepoRoot $repoRoot

$models = @(
    [PSCustomObject]@{
        name = "0.8b"
        cuda_variant = "0p8b"
        qwen_exe = Resolve-RepoPath -Path $Qwen08Exe -RepoRoot $repoRoot
        qwen_model_dir = Resolve-RepoPath -Path $Qwen08ModelDir -RepoRoot $repoRoot
        qwen_build_dir = "build-0p8b-bench"
        llama_gguf = Resolve-RepoPath -Path $Llama08Gguf -RepoRoot $repoRoot
    },
    [PSCustomObject]@{
        name = "4b"
        cuda_variant = "4b"
        qwen_exe = Resolve-RepoPath -Path $Qwen4BExe -RepoRoot $repoRoot
        qwen_model_dir = Resolve-RepoPath -Path $Qwen4BModelDir -RepoRoot $repoRoot
        qwen_build_dir = "build-4b-bench"
        llama_gguf = Resolve-RepoPath -Path $Llama4BGguf -RepoRoot $repoRoot
    }
)

if ($BuildQwenVariants -and -not $SkipQwen) {
    foreach ($model in $models) {
        Invoke-Checked -Label ("Building qwen35x CUDA variant {0}" -f $model.name) -Script {
            & $buildScript -UseNinja -EnableCuda -Configuration Release -BuildDir $model.qwen_build_dir -Target qwen35x -CudaVariant $model.cuda_variant
        }
    }
}

$summaryRows = @()
$llamaBuildDone = $false
$promptFilesByContext = @{}
foreach ($context in $Contexts) {
    $promptFilesByContext[$context] = New-PromptFileForContext -SourcePath $resolvedPromptSourceFile -SourceTokens $PromptSourceTokens -ContextTokens $context -OutDir $resolvedOutDir
}

if ($SummarizeOnly) {
    $summaryRows += Import-ExistingQwenRows -Models $models -Contexts $Contexts -OutDir $resolvedOutDir -MaxNewTokens $MaxNewTokens
    $summaryRows += Import-ExistingLlamaRows -Models $models -Contexts $Contexts -OutDir $resolvedOutDir -MaxNewTokens $MaxNewTokens

    if ($summaryRows.Count -eq 0) {
        throw "No existing benchmark output files were found to summarize in $resolvedOutDir."
    }

    $compactRows = Export-CompactComparisonCsvs -Rows $summaryRows -OutDir $resolvedOutDir -SummaryCsvOut $resolvedSummaryCsvOut
    Write-Host "Benchmark matrix summary written to: $resolvedSummaryCsvOut" -ForegroundColor Green
    Write-Host "Per-context comparison CSVs written to: $resolvedOutDir\\comparison-ctx*.csv" -ForegroundColor Green
    $compactRows |
        Format-Table model, implementation, ctx, prefill_tokens_per_second, generation_tokens_per_second -AutoSize
    exit 0
}

foreach ($model in $models) {
    if (-not $SkipQwen -and -not (Test-Path -LiteralPath $model.qwen_exe)) {
        throw "qwen35x executable for $($model.name) not found: $($model.qwen_exe). Re-run with -BuildQwenVariants or pass the executable path."
    }
    if (-not $SkipQwen -and -not (Test-Path -LiteralPath $model.qwen_model_dir)) {
        throw "HF model directory for $($model.name) not found: $($model.qwen_model_dir)"
    }
    if (-not $SkipLlama -and -not (Test-Path -LiteralPath $model.llama_gguf)) {
        throw "llama.cpp GGUF for $($model.name) not found: $($model.llama_gguf)"
    }

    foreach ($context in $Contexts) {
        if ($context -lt 1) {
            throw "Contexts must all be >= 1."
        }
        $promptTokensCsv = New-PromptTokenCsv -Count $context -Token $TokenId
        $promptFile = [string]$promptFilesByContext[$context]
        $safeModel = $model.name -replace '[^A-Za-z0-9]+', ''
        $labelBase = ("{0}-ctx{1}-gen{2}" -f $safeModel, $context, $MaxNewTokens)

        if (-not $SkipQwen) {
            $qwenPrefillCsv = Join-Path $resolvedOutDir ("qwen35x-{0}-ctx{1}-prefill.csv" -f $safeModel, $context)
            Invoke-Checked -Label ("qwen35x {0} prefill ctx={1}" -f $model.name, $context) -Script {
                & $qwenBench `
                    -Executable $model.qwen_exe `
                    -HFModelDir $model.qwen_model_dir `
                    -Modes gpu-f32 `
                    -PromptMode prompt-file `
                    -PromptName ("{0}_prefill" -f $labelBase) `
                    -PromptFile $promptFile `
                    -Runs $Runs `
                    -WarmupRuns $WarmupRuns `
                    -MaxNewTokens 0 `
                    -MaxContext $context `
                    -PrefillOnly `
                    -CsvOut $qwenPrefillCsv `
                    -RunLabel ("qwen35x-{0}-prefill" -f $labelBase)
            }
            $summaryRows += Import-QwenRows -CsvPath $qwenPrefillCsv -Model $model.name -ContextTokens $context -Workload "prefill"

            $qwenGenCsv = Join-Path $resolvedOutDir ("qwen35x-{0}-ctx{1}-gen{2}.csv" -f $safeModel, $context, $MaxNewTokens)
            Invoke-Checked -Label ("qwen35x {0} generation ctx={1}" -f $model.name, $context) -Script {
                & $qwenBench `
                    -Executable $model.qwen_exe `
                    -HFModelDir $model.qwen_model_dir `
                    -Modes gpu-f32 `
                    -PromptMode prompt-file `
                    -PromptName ("{0}_generation" -f $labelBase) `
                    -PromptFile $promptFile `
                    -Runs $Runs `
                    -WarmupRuns $WarmupRuns `
                    -MaxNewTokens $MaxNewTokens `
                    -MaxContext ($context + $MaxNewTokens) `
                    -CsvOut $qwenGenCsv `
                    -RunLabel ("qwen35x-{0}-generation" -f $labelBase)
            }
            $summaryRows += Import-QwenRows -CsvPath $qwenGenCsv -Model $model.name -ContextTokens $context -Workload "generation"
        }

        if (-not $SkipLlama) {
            if (-not (Test-Path -LiteralPath $resolvedLlamaExe)) {
                if (-not $BuildLlama) {
                    throw "llama actual-run executable not found: $resolvedLlamaExe. Re-run with -BuildLlama or pass -LlamaExe."
                }
            }
            foreach ($flash in @("off", "on")) {
                $flashSuffix = if ($flash -eq "on") { "fa" } else { "no-fa" }
                if ($BuildLlama -and -not $llamaBuildDone) {
                    $llamaBuildArgs = @{
                        GgufOut = $model.llama_gguf
                        SkipConvert = $true
                        SkipBench = $true
                        UseNinja = $true
                        EnableCuda = $true
                        HFModelDir = $model.qwen_model_dir
                        BuildTargets = @("llama-bench", "llama-completion", "llama-cli")
                    }
                    if ($CleanLlamaBuild) {
                        $llamaBuildArgs.CleanBuild = $true
                    }
                    Invoke-Checked -Label "Building llama.cpp actual-run targets" -Script {
                        & $llamaBench @llamaBuildArgs
                    }
                    $llamaBuildDone = $true
                    if (-not (Test-Path -LiteralPath $resolvedLlamaExe)) {
                        throw "llama actual-run executable not found after build: $resolvedLlamaExe"
                    }
                }

                $llamaActualCsv = Join-Path $resolvedOutDir ("llama-{0}-{1}-ctx{2}-actual-gen{3}.csv" -f $safeModel, $flashSuffix, $context, $MaxNewTokens)
                $llamaActualOutDir = Join-Path $resolvedOutDir ("llama-{0}-{1}-ctx{2}-actual-gen{3}" -f $safeModel, $flashSuffix, $context, $MaxNewTokens)
                Remove-Item -LiteralPath $llamaActualCsv -Force -ErrorAction SilentlyContinue
                Invoke-Checked -Label ("llama.cpp {0} {1} actual generation ctx={2}" -f $model.name, $flashSuffix, $context) -Script {
                    $llamaActualArgs = @{
                        LlamaExe = $resolvedLlamaExe
                        Model = $model.llama_gguf
                        PromptFile = $promptFile
                        OutputDir = $llamaActualOutDir
                        CsvOut = $llamaActualCsv
                        RunLabel = ("llama-{0}-{1}-ctx{2}-actual-gen{3}" -f $safeModel, $flashSuffix, $context, $MaxNewTokens)
                        FlashAttention = $flash
                        MaxContext = ($context + $MaxNewTokens)
                        MaxNewTokens = $MaxNewTokens
                        GpuLayers = $LlamaGpuLayers
                        BatchSize = $LlamaBatchSize
                        UBatchSize = $LlamaUBatchSize
                        TimeoutSeconds = $LlamaTimeoutSeconds
                    }
                    if ($LlamaNoWarmup.IsPresent) {
                        $llamaActualArgs.NoWarmup = $true
                    }
                    & $llamaActualBench @llamaActualArgs
                }
                $summaryRows += Import-LlamaActualRows -CsvPath $llamaActualCsv -Model $model.name -ContextTokens $context
            }
        }
    }
}

if ($summaryRows.Count -eq 0) {
    throw "No benchmark rows were produced."
}

if ($SkipQwen) {
    $summaryRows += Import-ExistingQwenRows -Models $models -Contexts $Contexts -OutDir $resolvedOutDir -MaxNewTokens $MaxNewTokens
}
if ($SkipLlama) {
    $summaryRows += Import-ExistingLlamaRows -Models $models -Contexts $Contexts -OutDir $resolvedOutDir -MaxNewTokens $MaxNewTokens
}

$compactRows = Export-CompactComparisonCsvs -Rows $summaryRows -OutDir $resolvedOutDir -SummaryCsvOut $resolvedSummaryCsvOut
Write-Host "Benchmark matrix summary written to: $resolvedSummaryCsvOut" -ForegroundColor Green
Write-Host "Per-context comparison CSVs written to: $resolvedOutDir\\comparison-ctx*.csv" -ForegroundColor Green
$compactRows |
    Format-Table model, implementation, ctx, prefill_tokens_per_second, generation_tokens_per_second -AutoSize
