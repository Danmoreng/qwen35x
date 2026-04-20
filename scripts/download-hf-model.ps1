[CmdletBinding()]
param(
    [string]$Repo = "Qwen/Qwen3.5-0.8B",
    [string]$OutDir = "models\qwen3.5-0.8b",
    [switch]$Full,
    [switch]$InstallDeps
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent $ScriptDir
$Downloader = Join-Path $ScriptDir "hf\download_model.py"

if (-not (Test-Path $Downloader)) {
    throw "Downloader script not found at $Downloader"
}

$ResolvedOutDir = if ([System.IO.Path]::IsPathRooted($OutDir)) {
    $OutDir
} else {
    Join-Path $RepoRoot $OutDir
}

if ($InstallDeps) {
    Write-Host "Installing/updating huggingface_hub (<1.0 for transformers compatibility)..." -ForegroundColor Cyan
    python -m pip install --upgrade "huggingface_hub<1.0"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install huggingface_hub."
    }
} else {
    python -c "import huggingface_hub" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "huggingface_hub is not installed. Re-run with -InstallDeps."
    }
}

$args = @(
    $Downloader,
    "--repo", $Repo,
    "--dest", $ResolvedOutDir
)
if ($Full) {
    $args += "--full"
}

Write-Host "Downloading $Repo into $ResolvedOutDir" -ForegroundColor Cyan
python @args
if ($LASTEXITCODE -ne 0) {
    throw "Model download failed."
}

Write-Host "Model download complete." -ForegroundColor Green
