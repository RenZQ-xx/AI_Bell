param(
    [ValidateRange(1, 86400)][int]$PollSeconds = 30,
    [ValidateRange(1, 86400)][int]$SnapshotSeconds = 3600,
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
$run = Join-Path $PSScriptRoot 'runs\catalog_reproduction\v1_s4222027'
$statusPath = Join-Path $run 'status.json'

function Read-SharedJson([string]$Path) {
    $share = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, $share)
    $reader = $null
    try {
        $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8)
        return ($reader.ReadToEnd() | ConvertFrom-Json)
    } finally {
        if ($reader) { $reader.Dispose() } else { $stream.Dispose() }
    }
}

$launch = Get-ChildItem -LiteralPath $run -Filter 'launch_*.json' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $launch) { throw 'No launch record is available.' }
$logs = Read-SharedJson $launch.FullName
$lastSnapshot = [DateTime]::UtcNow
while ($true) {
    try { $s = Read-SharedJson $statusPath } catch {
        if ($Once) { throw }
        Start-Sleep -Seconds 5
        continue
    }
    if ($Once) {
        $s | ConvertTo-Json -Depth 8
        exit 0
    }
    if ($s.state -in @('COMPLETED', 'FAILED', 'BATCH_LIMIT')) {
        $s | ConvertTo-Json -Depth 8
        Get-Content -LiteralPath $logs.stdout_log -Tail 12
        Get-Content -LiteralPath $logs.stderr_log -Tail 30
        if ($s.state -eq 'COMPLETED') { exit 0 }
        exit 1
    }
    $worker = Get-Process -Id $s.pid -ErrorAction SilentlyContinue
    if (-not $worker -or $worker.ProcessName -ne 'python') {
        Start-Sleep -Seconds 5
        $s = Read-SharedJson $statusPath
        if ($s.state -notin @('COMPLETED', 'FAILED', 'BATCH_LIMIT')) {
            Write-Output 'WORKER_DISAPPEARED: last saved state follows; no restart attempted.'
        }
        $s | ConvertTo-Json -Depth 8
        Get-Content -LiteralPath $logs.stderr_log -Tail 30
        if ($s.state -eq 'COMPLETED') { exit 0 }
        exit 1
    }
    if (([DateTime]::UtcNow - $lastSnapshot).TotalSeconds -ge $SnapshotSeconds) {
        [pscustomobject]@{
            state = $s.state; phase = $s.phase; iterations = $s.iterations_completed
            classes = $s.covered_catalog_class_count
            completed_nodes = $s.completed_nonempty_node_count
            elapsed_seconds = $s.elapsed_seconds; pid = $s.pid
            working_set_bytes = $worker.WorkingSet64; cpu_seconds = $worker.CPU
        } | ConvertTo-Json -Compress
        $lastSnapshot = [DateTime]::UtcNow
    }
    Start-Sleep -Seconds $PollSeconds
}
