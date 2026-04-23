[CmdletBinding()]
param(
    [string]$PythonExe = "python",
    [string]$VenvPath = ".venv-hf-parity",
    [string]$TransformersSpec = "git+https://github.com/huggingface/transformers.git",
    [switch]$NoVenv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $scriptDir
$requirements = Join-Path $repoRoot "scripts\hf\requirements-transformers-parity.txt"

if (-not (Test-Path -LiteralPath $requirements)) {
    throw "Requirements file not found: $requirements"
}

if ($NoVenv.IsPresent) {
    $resolvedPython = $PythonExe
} else {
    $resolvedVenv = if ([System.IO.Path]::IsPathRooted($VenvPath)) {
        $VenvPath
    } else {
        Join-Path $repoRoot $VenvPath
    }

    if (-not (Test-Path -LiteralPath $resolvedVenv)) {
        Write-Host "Creating Python venv: $resolvedVenv" -ForegroundColor Cyan
        & $PythonExe -m venv $resolvedVenv
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create Python venv."
        }
    }

    $resolvedPython = Join-Path $resolvedVenv "Scripts\python.exe"
}

Write-Host "Using Python: $resolvedPython" -ForegroundColor Cyan
& $resolvedPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed."
}

& $resolvedPython -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) {
    throw "Transformers parity dependency install failed."
}

Write-Host "Installing Transformers: $TransformersSpec" -ForegroundColor Cyan
& $resolvedPython -m pip install --upgrade $TransformersSpec
if ($LASTEXITCODE -ne 0) {
    throw "Transformers install failed."
}

Write-Host "Transformers parity dependencies are installed." -ForegroundColor Green
Write-Host "Use: $resolvedPython scripts\hf\transformers_inference.py --help" -ForegroundColor Green
