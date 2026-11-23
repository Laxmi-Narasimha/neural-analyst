param(
    [switch]$Celery,
    [switch]$Minio,
    [switch]$Detach
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$BackendEnv = Join-Path $RepoRoot "ai-data-analyst\backend\.env"
$BackendEnvExample = Join-Path $RepoRoot "ai-data-analyst\backend\.env.example"

Push-Location $RepoRoot

try {
    if (-not (Test-Path $BackendEnv) -and (Test-Path $BackendEnvExample)) {
        Copy-Item $BackendEnvExample $BackendEnv
        Write-Host "Created $BackendEnv from .env.example — set LLM_MODEL and your API key."
    }

    $files = @("docker-compose.yml")
    if ($Celery) {
        $files += "docker-compose.celery.yml"
    }
    if ($Minio) {
        $files += "docker-compose.minio.yml"
    }

    $args = @("compose")
    foreach ($f in $files) {
        $args += @("-f", $f)
    }
    $args += @("up", "--build")
    if ($Detach) {
        $args += "-d"
    }

    Write-Host ""
    Write-Host "Neural Analyst self-host"
    Write-Host "  UI:  http://localhost:3000"
    Write-Host "  API: http://localhost:8000/docs"
    Write-Host "  BYOK setup: http://localhost:3000/setup-keys"
    Write-Host "  Keys/compute: edit ai-data-analyst/backend/.env (runs on YOUR RAM/CPU)"
    Write-Host ""

    docker @args
} finally {
    Pop-Location
}