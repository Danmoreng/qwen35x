[CmdletBinding()]
param(
    [int]$Port = 8787,
    [string]$PythonExe = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$benchDir = Join-Path $repoRoot "tools\webgpu-transformers-bench"

if (-not (Test-Path -LiteralPath (Join-Path $benchDir "index.html"))) {
    throw "WebGPU benchmark page not found: $benchDir"
}

Write-Host "Serving Qwen3.5 WebGPU benchmark from: $benchDir" -ForegroundColor Green
Write-Host "Open: http://127.0.0.1:$Port/" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow

Push-Location $benchDir
try {
    & $PythonExe -m http.server $Port --bind 127.0.0.1
} finally {
    Pop-Location
}
