$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$BackendDir = Join-Path $RepoRoot "ai-data-analyst\\backend"
$FrontendDir = Join-Path $RepoRoot "ai-data-analyst\\frontend"

Write-Host "RepoRoot: $RepoRoot"

Write-Host ""
Write-Host "== Backend deps =="
Push-Location $BackendDir
try {
    if (!(Test-Path ".\\.venv")) {
        Write-Host "Creating venv at $BackendDir\\.venv ..."
        python -m venv .venv
    }

    .\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
    .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "== Frontend deps =="
Push-Location $FrontendDir
try {
    npm ci
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Done."

