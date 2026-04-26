[CmdletBinding()]
param (
    [switch]$Clean,
    [switch]$UseNinja,
    [switch]$EnableCuda,
    [switch]$BuildAll,
    [switch]$RunSmokeTest,
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [string]$BuildDir = "build",
    [string]$Target = "qwen35x",
    [string]$Profile = "configs/qwen3_5_0_8b.profile.json",
    [int]$SmVersion = 120,
    [string]$CudaArchitectures = "native",
    [ValidateSet("0p8b", "4b")]
    [string]$CudaVariant = "0p8b",
    [Alias("LuceBlockSize")]
    [int]$KernelBlockSize = 256,
    [Alias("LuceNumBlocks")]
    [int]$KernelNumBlocks = 82
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

$resolvedBuildDir = if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    $BuildDir
} else {
    Join-Path $RepoRoot $BuildDir
}

$resolvedProfile = if ([System.IO.Path]::IsPathRooted($Profile)) {
    $Profile
} else {
    Join-Path $RepoRoot $Profile
}

Import-VSEnv

if ($Clean -and (Test-Path $resolvedBuildDir)) {
    Write-Host "Cleaning build directory: $resolvedBuildDir" -ForegroundColor Yellow
    Remove-Item -Path $resolvedBuildDir -Recurse -Force
}
New-Item -Path $resolvedBuildDir -ItemType Directory -Force | Out-Null

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

$configuredGenerator = Get-ConfiguredGenerator -buildDir $resolvedBuildDir
$expectedGenerator = if ($isNinja) { "Ninja" } else { "Visual Studio 17 2022" }
if ($configuredGenerator -and $configuredGenerator -ne $expectedGenerator) {
    Write-Host "Build directory generator mismatch: '$configuredGenerator' -> '$expectedGenerator'" -ForegroundColor Yellow
    Remove-Item -Path $resolvedBuildDir -Recurse -Force
    New-Item -Path $resolvedBuildDir -ItemType Directory -Force | Out-Null
}

$cudaFlag = if ($EnableCuda) { "ON" } else { "OFF" }

$configureArgs = @(
    "-S", $RepoRoot,
    "-B", $resolvedBuildDir
) + $GeneratorArgs + @(
    "-DCMAKE_CXX_STANDARD=20",
    "-DQWEN35X_ENABLE_CUDA=$cudaFlag",
    "-DQWEN35X_CUDA_ARCHITECTURES=$CudaArchitectures",
    "-DQWEN35X_CUDA_VARIANT=$CudaVariant",
    "-DQWEN35X_KERNEL_BENCH_BLOCK_SIZE=$KernelBlockSize",
    "-DQWEN35X_KERNEL_BENCH_NUM_BLOCKS=$KernelNumBlocks"
)

if ($isNinja) {
    $configureArgs += "-DCMAKE_BUILD_TYPE=$Configuration"
}

Write-Host "Configuring CMake in: $resolvedBuildDir" -ForegroundColor Cyan
& cmake @configureArgs
if ($LASTEXITCODE -ne 0) {
    throw "CMake configuration failed."
}

$resolvedTarget = if ($BuildAll) {
    if ($isNinja) { "all" } else { "ALL_BUILD" }
} else {
    $Target
}

Write-Host "Building target: $resolvedTarget ($Configuration)" -ForegroundColor Cyan
$buildArgs = @("--build", $resolvedBuildDir, "--target", $resolvedTarget, "--parallel")
if (-not $isNinja) {
    $buildArgs += @("--config", $Configuration)
}
& cmake @buildArgs
if ($LASTEXITCODE -ne 0) {
    throw "Build failed."
}

$exePath = Find-FirstExisting @(
    (Join-Path $resolvedBuildDir "$Configuration\qwen35x.exe"),
    (Join-Path $resolvedBuildDir "qwen35x.exe"),
    (Join-Path $resolvedBuildDir "bin\$Configuration\qwen35x.exe"),
    (Join-Path $resolvedBuildDir "bin\qwen35x.exe")
)

Write-Host "Build success." -ForegroundColor Green
if ($exePath) {
    Write-Host "Executable: $exePath"
} else {
    Write-Host "Executable not found in expected paths." -ForegroundColor Yellow
}

if ($RunSmokeTest) {
    if (-not $exePath) {
        throw "RunSmokeTest requested, but qwen35x.exe was not found."
    }
    if (-not (Test-Path $resolvedProfile)) {
        throw "RunSmokeTest requested, profile not found: $resolvedProfile"
    }

    Write-Host "Running smoke test..." -ForegroundColor Cyan
    & $exePath --profile $resolvedProfile --sm $SmVersion
    if ($LASTEXITCODE -ne 0) {
        throw "Smoke test failed."
    }
    Write-Host "Smoke test passed." -ForegroundColor Green
}
