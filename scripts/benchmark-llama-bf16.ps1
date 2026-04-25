[CmdletBinding()]
param (
    [string]$LlamaCppDir = "third_party/reference/llama.cpp",
    [string]$HFModelDir = "models/qwen3.5-0.8b",
    [string]$GgufOut = "models/gguf/qwen3.5-0.8b-bf16.gguf",
    [ValidateSet("bf16", "f16", "f32", "q8_0", "auto")]
    [string]$OutType = "bf16",
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [switch]$UseNinja,
    [switch]$EnableCuda = $true,
    [switch]$InstallConvertDeps,
    [switch]$CleanBuild,
    [switch]$SkipConvert,
    [switch]$SkipBuild,
    [switch]$SkipBench,
    [string]$BenchOut = "benchmarks/llama-bench/qwen3.5-0.8b-bf16.json",
    [int]$PromptTokens = 512,
    [int]$GenTokens = 128,
    [int]$Repetitions = 5,
    [int]$NGpuLayers = 99,
    [int]$BatchSize = 2048,
    [int]$UBatchSize = 512,
    [switch]$FlashAttention,
    [string[]]$BuildTargets = @("llama-bench")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Import-VSEnv {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        return
    }

    $vswhere = Join-Path ${Env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe not found. Install Visual Studio Build Tools with C++ workload."
    }

    $vsroot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if ([string]::IsNullOrWhiteSpace($vsroot)) {
        throw "Visual Studio C++ toolchain not found."
    }

    $vcvars = Join-Path $vsroot "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) {
        throw "vcvars64.bat not found at $vcvars"
    }

    Write-Host "Importing Visual Studio environment from: $vcvars" -ForegroundColor Cyan
    $envDump = cmd /s /c "`"$vcvars`" > nul && set"
    foreach ($line in $envDump) {
        if ($line -match "^(.*?)=(.*)$") {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }
}

function Get-ConfiguredGenerator([string]$buildDir) {
    $cachePath = Join-Path $buildDir "CMakeCache.txt"
    if (-not (Test-Path $cachePath)) {
        return $null
    }

    $line = Select-String -Path $cachePath -Pattern "^CMAKE_GENERATOR:INTERNAL=(.+)$" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($line) {
        return $line.Matches[0].Groups[1].Value
    }
    return $null
}

function Find-FirstExisting([string[]]$paths) {
    foreach ($path in $paths) {
        if ($path -and (Test-Path $path)) {
            return $path
        }
    }
    return $null
}

$ScriptDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent $ScriptDir

$ResolvedLlamaCppDir = if ([System.IO.Path]::IsPathRooted($LlamaCppDir)) { $LlamaCppDir } else { Join-Path $RepoRoot $LlamaCppDir }
$ResolvedHFModelDir = if ([System.IO.Path]::IsPathRooted($HFModelDir)) { $HFModelDir } else { Join-Path $RepoRoot $HFModelDir }
$ResolvedGgufOut = if ([System.IO.Path]::IsPathRooted($GgufOut)) { $GgufOut } else { Join-Path $RepoRoot $GgufOut }
$ResolvedBenchOut = if ([System.IO.Path]::IsPathRooted($BenchOut)) { $BenchOut } else { Join-Path $RepoRoot $BenchOut }

if (-not (Test-Path (Join-Path $ResolvedLlamaCppDir "CMakeLists.txt"))) {
    throw "llama.cpp directory not found or invalid: $ResolvedLlamaCppDir"
}
if (-not (Test-Path $ResolvedHFModelDir)) {
    throw "HF model directory not found: $ResolvedHFModelDir"
}

$ResolvedBuildDir = Join-Path $ResolvedLlamaCppDir "build-qwen35x"
$ConvertScript = Join-Path $ResolvedLlamaCppDir "convert_hf_to_gguf.py"
if (-not (Test-Path $ConvertScript)) {
    throw "convert_hf_to_gguf.py not found in llama.cpp dir: $ResolvedLlamaCppDir"
}

if ($InstallConvertDeps) {
    $Req = Join-Path $ResolvedLlamaCppDir "requirements/requirements-convert_hf_to_gguf.txt"
    if (-not (Test-Path $Req)) {
        throw "Conversion requirements file not found: $Req"
    }
    Write-Host "Installing conversion dependencies from $Req" -ForegroundColor Cyan
    python -m pip install -r $Req
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install conversion dependencies."
    }
}

if (-not $SkipConvert) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $ResolvedGgufOut) -Force | Out-Null
    Write-Host "Converting HF model to GGUF: $ResolvedGgufOut ($OutType)" -ForegroundColor Cyan
    python $ConvertScript $ResolvedHFModelDir --outfile $ResolvedGgufOut --outtype $OutType
    if ($LASTEXITCODE -ne 0) {
        throw "HF -> GGUF conversion failed."
    }
}

if (-not (Test-Path $ResolvedGgufOut)) {
    throw "GGUF file not found: $ResolvedGgufOut"
}

if (-not $SkipBuild) {
    Import-VSEnv

    if ($CleanBuild -and (Test-Path $ResolvedBuildDir)) {
        Write-Host "Cleaning llama.cpp build directory: $ResolvedBuildDir" -ForegroundColor Yellow
        Remove-Item -Path $ResolvedBuildDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $ResolvedBuildDir -Force | Out-Null

    $GeneratorArgs = @()
    $isNinja = $false
    if ($UseNinja) {
        if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
            throw "Ninja was requested with -UseNinja but was not found on PATH."
        }
        $GeneratorArgs += @("-G", "Ninja")
        $isNinja = $true
        Write-Host "Generator: Ninja"
    } else {
        $GeneratorArgs += @("-G", "Visual Studio 17 2022", "-A", "x64")
        Write-Host "Generator: Visual Studio 17 2022"
    }

    $configuredGenerator = Get-ConfiguredGenerator -buildDir $ResolvedBuildDir
    $expectedGenerator = if ($isNinja) { "Ninja" } else { "Visual Studio 17 2022" }
    if ($configuredGenerator -and $configuredGenerator -ne $expectedGenerator) {
        Write-Host "Build directory generator mismatch: '$configuredGenerator' -> '$expectedGenerator'" -ForegroundColor Yellow
        Remove-Item -Path $ResolvedBuildDir -Recurse -Force
        New-Item -Path $ResolvedBuildDir -ItemType Directory -Force | Out-Null
    }

    $cudaFlag = if ($EnableCuda) { "ON" } else { "OFF" }
    $ConfigureArgs = @(
        "-S", $ResolvedLlamaCppDir,
        "-B", $ResolvedBuildDir
    ) + $GeneratorArgs + @(
        "-DGGML_CUDA=$cudaFlag"
    )
    if ($isNinja) {
        $ConfigureArgs += "-DCMAKE_BUILD_TYPE=$Configuration"
    }

    Write-Host "Configuring llama.cpp in: $ResolvedBuildDir" -ForegroundColor Cyan
    & cmake @ConfigureArgs
    if ($LASTEXITCODE -ne 0) {
        throw "llama.cpp CMake configuration failed."
    }

    $BuildArgs = @("--build", $ResolvedBuildDir)
    foreach ($target in $BuildTargets) {
        $BuildArgs += @("--target", $target)
    }
    $BuildArgs += "--parallel"
    if (-not $isNinja) {
        $BuildArgs += @("--config", $Configuration)
    }

    Write-Host ("Building target(s): {0} ({1})" -f ($BuildTargets -join ", "), $Configuration) -ForegroundColor Cyan
    & cmake @BuildArgs
    if ($LASTEXITCODE -ne 0) {
        throw "llama.cpp build failed."
    }
}

$BenchExe = Find-FirstExisting @(
    (Join-Path $ResolvedBuildDir "$Configuration\bin\llama-bench.exe"),
    (Join-Path $ResolvedBuildDir "bin\$Configuration\llama-bench.exe"),
    (Join-Path $ResolvedBuildDir "bin\llama-bench.exe"),
    (Join-Path $ResolvedBuildDir "$Configuration\llama-bench.exe"),
    (Join-Path $ResolvedBuildDir "llama-bench.exe")
)

if (-not $SkipBench) {
    if (-not $BenchExe) {
        throw "llama-bench executable not found in build output."
    }

    New-Item -ItemType Directory -Path (Split-Path -Parent $ResolvedBenchOut) -Force | Out-Null

    $BenchArgs = @(
        "--model", $ResolvedGgufOut,
        "--n-gpu-layers", $NGpuLayers,
        "--batch-size", $BatchSize,
        "--ubatch-size", $UBatchSize,
        "--n-prompt", $PromptTokens,
        "--n-gen", $GenTokens,
        "--repetitions", $Repetitions,
        "--output", "json"
    )
    if ($FlashAttention) {
        $BenchArgs += @("--flash-attn", "1")
    }

    Write-Host "Running llama-bench..." -ForegroundColor Cyan
    & $BenchExe @BenchArgs | Tee-Object -FilePath $ResolvedBenchOut
    if ($LASTEXITCODE -ne 0) {
        throw "llama-bench failed."
    }

    Write-Host "Benchmark output written to: $ResolvedBenchOut" -ForegroundColor Green
} else {
    Write-Host "Setup complete. Skipped benchmark run (-SkipBench)." -ForegroundColor Green
    if ($BenchExe) {
        Write-Host "Ready command:" -ForegroundColor Green
        Write-Host "  $BenchExe --model `"$ResolvedGgufOut`" --n-gpu-layers $NGpuLayers --batch-size $BatchSize --ubatch-size $UBatchSize --n-prompt $PromptTokens --n-gen $GenTokens --repetitions $Repetitions --output json" -ForegroundColor Green
    } else {
        Write-Host "llama-bench executable not found yet. Build first by re-running without -SkipBuild." -ForegroundColor Yellow
    }
}
