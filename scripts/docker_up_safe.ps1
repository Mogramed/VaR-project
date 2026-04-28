param(
    [switch]$NoBuild,
    [switch]$FollowLogs,
    [switch]$NoPrePull,
    [switch]$WithObservability,
    [int]$MaxRetries = 4,
    [int]$InitialRetryDelaySeconds = 3,
    [int]$ComposeParallelLimit = 3
)

$ErrorActionPreference = "Stop"

# Work around intermittent Docker Compose v5 monitor panics on Windows hosts.
$env:COMPOSE_MENU = "false"
$env:COMPOSE_PROGRESS = "plain"
$env:COMPOSE_PARALLEL_LIMIT = [Math]::Max(1, $ComposeParallelLimit).ToString()

function Test-IsTransientDockerError {
    param([string]$Message)

    if (-not $Message) {
        return $false
    }

    $transientPatterns = @(
        "tls: bad record MAC",
        "failed to fetch oauth token",
        "error reading from server: EOF",
        "i/o timeout",
        "context deadline exceeded",
        "connection reset by peer",
        "TLS handshake timeout",
        "temporary failure in name resolution",
        "net/http: request canceled",
        "429 Too Many Requests"
    )

    $content = $Message.ToLowerInvariant()
    foreach ($pattern in $transientPatterns) {
        if ($content.Contains($pattern.ToLowerInvariant())) {
            return $true
        }
    }

    return $false
}

function Invoke-DockerCommandWithRetry {
    param(
        [string]$StepName,
        [string[]]$CommandArgs
    )

    $maxAttempts = [Math]::Max(1, $MaxRetries)
    $attempt = 1

    while ($attempt -le $maxAttempts) {
        Write-Host "[$StepName] Attempt $attempt/$maxAttempts -> docker $($CommandArgs -join ' ')" -ForegroundColor DarkCyan
        $captured = @()
        docker @CommandArgs 2>&1 | ForEach-Object {
            $line = "$_"
            $captured += $line
            Write-Host $line
        }

        if ($LASTEXITCODE -eq 0) {
            return
        }

        $fullOutput = ($captured -join [Environment]::NewLine)
        $isTransient = Test-IsTransientDockerError -Message $fullOutput
        if (-not $isTransient -or $attempt -ge $maxAttempts) {
            throw "[$StepName] docker $($CommandArgs -join ' ') failed (exit $LASTEXITCODE)."
        }

        $delay = [Math]::Min(45, [int]($InitialRetryDelaySeconds * [Math]::Pow(2, $attempt - 1)))
        Write-Host "[$StepName] Transient Docker/network error detected. Retrying in $delay second(s)..." -ForegroundColor Yellow
        Start-Sleep -Seconds $delay
        $attempt += 1
    }
}

Write-Host "Starting Docker stack with safe defaults..." -ForegroundColor Cyan
Write-Host "Compose parallel limit: $env:COMPOSE_PARALLEL_LIMIT" -ForegroundColor DarkGray
Write-Host "Observability profile: $($WithObservability.IsPresent)" -ForegroundColor DarkGray

if (-not $NoPrePull) {
    $baseImages = @(
        "nginx:1.27-alpine",
        "node:22-bookworm-slim",
        "python:3.12-slim",
        "redis:7-alpine",
        "postgres:16-alpine"
    )
    if ($WithObservability) {
        $baseImages += @(
            "prom/prometheus:v2.54.1",
            "prom/alertmanager:v0.28.0",
            "grafana/grafana:11.6.0"
        )
    }
    foreach ($image in $baseImages) {
        Invoke-DockerCommandWithRetry -StepName "Pre-pull $image" -CommandArgs @("pull", $image)
    }
}

$composeArgs = @("compose", "up", "-d")
if ($WithObservability) {
    $composeArgs = @("compose", "--profile", "observability", "up", "-d")
}
if (-not $NoBuild) {
    $composeArgs += "--build"
}
Invoke-DockerCommandWithRetry -StepName "Compose up" -CommandArgs $composeArgs

Write-Host ""
Write-Host "Current service status:" -ForegroundColor Cyan
Invoke-DockerCommandWithRetry -StepName "Compose ps" -CommandArgs @("compose", "ps")

if ($FollowLogs) {
    Write-Host ""
    Write-Host "Following logs (Ctrl+C to stop tailing, services stay up)..." -ForegroundColor Cyan
    docker compose logs -f
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Stack started. Use 'docker compose logs -f' to follow logs." -ForegroundColor Green
