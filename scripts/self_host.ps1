param(
    [switch]$Celery,
    [switch]$Minio,
    [switch]$Detach
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot

try {
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

    docker @args
} finally {
    Pop-Location
}
