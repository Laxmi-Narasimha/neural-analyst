$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$BackendDir = Join-Path $RepoRoot "ai-data-analyst\\backend"
$FrontendDir = Join-Path $RepoRoot "ai-data-analyst\\frontend"

$BackendCmd = @"
cd `"$BackendDir`"
`$env:DATABASE_URL = "sqlite+aiosqlite:///./dev.db"
if (!(Test-Path ".\\.venv")) { python -m venv .venv }
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --reload --port 8000
"@

$FrontendCmd = @"
cd `"$FrontendDir`"
npm ci
`$env:NEXT_PUBLIC_API_URL = "http://localhost:8000/api/v1"
npm run dev
"@

Write-Host "Starting backend + frontend in separate PowerShell windows..."

Start-Process powershell -ArgumentList @("-NoExit", "-Command", $BackendCmd) | Out-Null
Start-Process powershell -ArgumentList @("-NoExit", "-Command", $FrontendCmd) | Out-Null

Write-Host ""
Write-Host "Backend:  http://localhost:8000"
Write-Host "Frontend: http://localhost:3000"

