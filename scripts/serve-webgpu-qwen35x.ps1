[CmdletBinding()]
param(
    [int]$Port = 8790,
    [string]$PythonExe = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$appDir = Join-Path $repoRoot "tools\webgpu-qwen35x"
$python = (Get-Command $PythonExe -ErrorAction Stop).Source

if (-not (Test-Path -LiteralPath (Join-Path $appDir "index.html"))) {
    throw "WebGPU app not found: $appDir"
}

Write-Host "Serving qwen35x WebGPU runtime from repo root: $repoRoot" -ForegroundColor Green
Write-Host "Open: http://127.0.0.1:$Port/tools/webgpu-qwen35x/" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow

& $python -m http.server $Port --bind 127.0.0.1 --directory $repoRoot
