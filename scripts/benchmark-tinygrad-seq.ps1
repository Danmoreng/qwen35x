[CmdletBinding()]
param(
    [string]$PythonExe = "C:\Users\User\AppData\Local\Python\bin\python.exe",
    [string]$TinygradRoot = "third_party/reference/tinygrad",
    [string]$RunnerScript = "scripts/tinygrad/benchmark_tinygrad_inference.py",
    [string]$GgufModel = "models/gguf/qwen3.5-0.8b-bf16.gguf",
    [string]$CsvOut = "benchmarks/tinygrad-inference-seq.csv",
    [string]$RunLabel = "",
    [ValidateSet("CUDA:PTX", "CUDA", "CL", "CPU")]
    [string]$Device = "CUDA:PTX",
    [ValidateSet("chat-user", "prompt-text")]
    [string]$PromptMode = "chat-user",
    [string]$PromptName = "chat_short_joke",
    [string]$PromptText = "Tell me a short joke.",
    [int]$Runs = 3,
    [int]$WarmupRuns = 1,
    [int]$MaxNewTokens = 128,
    [int]$MaxContext = 256,
    [double]$Temperature = 0.0,
    [int64]$Seed = 123,
    [switch]$StopOnEos,
    [string]$CudaDriverDll = "C:\Windows\System32\nvcuda.dll",
    [string]$CudaToolkitRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
    [string]$NvrtcDll = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64\nvrtc64_130_0.dll",
    [string]$NvJitLinkDll = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64\nvJitLink_130_0.dll"
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

function Prepend-PathIfMissing {
    param([Parameter(Mandatory = $true)][string]$PathToAdd)
    if ([string]::IsNullOrWhiteSpace($PathToAdd)) {
        return
    }
    if (-not (Test-Path $PathToAdd)) {
        return
    }

    $parts = @()
    if (-not [string]::IsNullOrEmpty($env:PATH)) {
        $parts = $env:PATH -split ';'
    }
    if ($parts -contains $PathToAdd) {
        return
    }
    $env:PATH = "$PathToAdd;$env:PATH"
}

function Set-TinygradEnvironment {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$ResolvedTinygradRoot,
        [Parameter(Mandatory = $true)][string]$Device,
        [Parameter(Mandatory = $true)][string]$CudaDriverDll,
        [Parameter(Mandatory = $true)][string]$CudaToolkitRoot,
        [Parameter(Mandatory = $true)][string]$NvrtcDll,
        [Parameter(Mandatory = $true)][string]$NvJitLinkDll
    )

    $cacheRoot = Join-Path $RepoRoot "build\tinygrad-cache"
    New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null
    $env:XDG_CACHE_HOME = $cacheRoot
    $env:CACHEDB = Join-Path $cacheRoot "cache.db"

    $env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $ResolvedTinygradRoot
    } else {
        "$ResolvedTinygradRoot;$env:PYTHONPATH"
    }

    $env:DEV = $Device

    if ($Device.StartsWith("CUDA")) {
        if (-not (Test-Path $CudaDriverDll)) {
            throw "CUDA driver DLL not found: $CudaDriverDll"
        }
        if (-not (Test-Path $CudaToolkitRoot)) {
            throw "CUDA toolkit root not found: $CudaToolkitRoot"
        }
        if (-not (Test-Path $NvrtcDll)) {
            throw "NVRTC DLL not found: $NvrtcDll"
        }
        if (-not (Test-Path $NvJitLinkDll)) {
            throw "nvJitLink DLL not found: $NvJitLinkDll"
        }

        $shimDir = Join-Path $RepoRoot "build\tinygrad-win-dll-shim"
        New-Item -ItemType Directory -Path $shimDir -Force | Out-Null
        $cudaShim = Join-Path $shimDir "cuda.dll"
        Copy-Item -LiteralPath $CudaDriverDll -Destination $cudaShim -Force

        $cudaBinDir = Join-Path $CudaToolkitRoot "bin\x64"
        Prepend-PathIfMissing -PathToAdd $shimDir
        Prepend-PathIfMissing -PathToAdd $cudaBinDir

        # tinygrad runtime compiler expects CUDA_PATH to be toolkit root.
        $env:CUDA_PATH = $CudaToolkitRoot
        $env:NVRTC_PATH = $NvrtcDll
        $env:NVJITLINK_PATH = $NvJitLinkDll
    }
}

function Invoke-TinygradRun {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string]$RunnerScript,
        [Parameter(Mandatory = $true)][string]$TinygradRoot,
        [Parameter(Mandatory = $true)][string]$ModelPath,
        [Parameter(Mandatory = $true)][string]$PromptMode,
        [Parameter(Mandatory = $true)][string]$PromptText,
        [Parameter(Mandatory = $true)][int]$MaxNewTokens,
        [Parameter(Mandatory = $true)][int]$MaxContext,
        [Parameter(Mandatory = $true)][double]$Temperature,
        [Parameter(Mandatory = $true)][int64]$Seed,
        [Parameter(Mandatory = $true)][string]$Device,
        [Parameter(Mandatory = $true)][bool]$StopOnEosEnabled,
        [Parameter(Mandatory = $true)][string]$ProfileJsonPath
    )

    $args = @(
        $RunnerScript,
        "--tinygrad-root", $TinygradRoot,
        "--model", $ModelPath,
        "--prompt-mode", $PromptMode,
        "--prompt-text", $PromptText,
        "--max-new-tokens", "$MaxNewTokens",
        "--max-context", "$MaxContext",
        "--temperature", (To-InvariantString $Temperature),
        "--seed", "$Seed",
        "--device", $Device,
        "--output-json", $ProfileJsonPath
    )
    if ($StopOnEosEnabled) {
        $args += "--stop-on-eos"
    }

    Write-Host ("Running tinygrad backend={0} prompt={1}" -f $Device, $PromptMode) -ForegroundColor Cyan
    $runOutput = & $PythonExe @args 2>&1
    foreach ($line in $runOutput) {
        Write-Host $line
    }
    if ($LASTEXITCODE -ne 0) {
        throw "tinygrad benchmark run failed (exit_code=$LASTEXITCODE)."
    }
    if (-not (Test-Path $ProfileJsonPath)) {
        throw "Missing tinygrad profile JSON: $ProfileJsonPath"
    }

    return Get-Content -Raw -LiteralPath $ProfileJsonPath | ConvertFrom-Json
}

$scriptDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $scriptDir

$resolvedPythonExe = Resolve-RepoPath -Path $PythonExe -RepoRoot $repoRoot
$resolvedTinygradRoot = Resolve-RepoPath -Path $TinygradRoot -RepoRoot $repoRoot
$resolvedRunnerScript = Resolve-RepoPath -Path $RunnerScript -RepoRoot $repoRoot
$resolvedModel = Resolve-RepoPath -Path $GgufModel -RepoRoot $repoRoot
$resolvedCsvOut = Resolve-RepoPath -Path $CsvOut -RepoRoot $repoRoot
$profileTmpDir = Join-Path $repoRoot "build\bench-profiles"

if (-not (Test-Path $resolvedPythonExe)) {
    throw "Python executable not found: $resolvedPythonExe"
}
if (-not (Test-Path $resolvedTinygradRoot)) {
    throw "tinygrad root not found: $resolvedTinygradRoot"
}
if (-not (Test-Path $resolvedRunnerScript)) {
    throw "Runner script not found: $resolvedRunnerScript"
}
if (-not (Test-Path $resolvedModel)) {
    throw "GGUF model not found: $resolvedModel"
}
if ($Runs -lt 1) {
    throw "Runs must be >= 1."
}
if ($WarmupRuns -lt 0) {
    throw "WarmupRuns must be >= 0."
}

New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedCsvOut) -Force | Out-Null
New-Item -ItemType Directory -Path $profileTmpDir -Force | Out-Null

Set-TinygradEnvironment `
    -RepoRoot $repoRoot `
    -ResolvedTinygradRoot $resolvedTinygradRoot `
    -Device $Device `
    -CudaDriverDll $CudaDriverDll `
    -CudaToolkitRoot $CudaToolkitRoot `
    -NvrtcDll $NvrtcDll `
    -NvJitLinkDll $NvJitLinkDll

Write-Host ("Sequential tinygrad benchmark start: device={0}, warmup={1}, runs={2}" -f $Device, $WarmupRuns, $Runs) -ForegroundColor Green
Write-Host ("CSV output: {0}" -f $resolvedCsvOut) -ForegroundColor Green

for ($warm = 1; $warm -le $WarmupRuns; ++$warm) {
    $warmProfile = Join-Path $profileTmpDir ("warmup_tinygrad_{0}_{1}_{2}.json" -f $Device.Replace(':', '_'), $warm, [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())
    try {
        $null = Invoke-TinygradRun `
            -PythonExe $resolvedPythonExe `
            -RunnerScript $resolvedRunnerScript `
            -TinygradRoot $resolvedTinygradRoot `
            -ModelPath $resolvedModel `
            -PromptMode $PromptMode `
            -PromptText $PromptText `
            -MaxNewTokens $MaxNewTokens `
            -MaxContext $MaxContext `
            -Temperature $Temperature `
            -Seed $Seed `
            -Device $Device `
            -StopOnEosEnabled $StopOnEos.IsPresent `
            -ProfileJsonPath $warmProfile
        Write-Host ("Warmup completed: run={0}/{1}" -f $warm, $WarmupRuns) -ForegroundColor DarkGray
    } finally {
        if (Test-Path $warmProfile) {
            Remove-Item -LiteralPath $warmProfile -Force
        }
    }
}

for ($runIndex = 1; $runIndex -le $Runs; ++$runIndex) {
    $profilePath = Join-Path $profileTmpDir ("run_tinygrad_{0}_{1}_{2}.json" -f $Device.Replace(':', '_'), $runIndex, [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())
    try {
        $profile = Invoke-TinygradRun `
            -PythonExe $resolvedPythonExe `
            -RunnerScript $resolvedRunnerScript `
            -TinygradRoot $resolvedTinygradRoot `
            -ModelPath $resolvedModel `
            -PromptMode $PromptMode `
            -PromptText $PromptText `
            -MaxNewTokens $MaxNewTokens `
            -MaxContext $MaxContext `
            -Temperature $Temperature `
            -Seed $Seed `
            -Device $Device `
            -StopOnEosEnabled $StopOnEos.IsPresent `
            -ProfileJsonPath $profilePath

        $row = [PSCustomObject]@{
            timestamp_utc      = [DateTime]::UtcNow.ToString("o")
            run_label          = $RunLabel
            mode               = "tinygrad-$Device"
            run_index          = $runIndex
            prompt_name        = $PromptName
            prompt_tokens      = [int]$profile.prompt_tokens
            generated_tokens   = [int]$profile.generated_tokens
            load_time_ms       = To-InvariantString $profile.load_time_ms
            decode_time_ms     = To-InvariantString $profile.decode_time_ms
            tokens_per_second  = To-InvariantString $profile.tokens_per_second
            backend            = [string]$profile.backend
        }

        if (Test-Path $resolvedCsvOut) {
            $row | Export-Csv -LiteralPath $resolvedCsvOut -NoTypeInformation -Append
        } else {
            $row | Export-Csv -LiteralPath $resolvedCsvOut -NoTypeInformation
        }

        Write-Host ("Recorded: run={0}/{1} tps={2}" -f $runIndex, $Runs, $row.tokens_per_second) -ForegroundColor Yellow
    } finally {
        if (Test-Path $profilePath) {
            Remove-Item -LiteralPath $profilePath -Force
        }
    }
}

Write-Host "tinygrad benchmark complete." -ForegroundColor Green
Write-Host ("CSV written to: {0}" -f $resolvedCsvOut) -ForegroundColor Green
