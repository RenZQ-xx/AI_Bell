param([switch]$Resume)

$ErrorActionPreference = 'Stop'
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$python = Join-Path $project '.venv\Scripts\python.exe'
$run = Join-Path $PSScriptRoot 'runs\catalog_reproduction\v1_s4222027'
$manifest = Join-Path $run 'manifest.json'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Project virtual environment is missing.'
}
if ($Resume -and -not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
    throw 'Resume requires an existing manifest.'
}
if (-not $Resume -and (Test-Path -LiteralPath $manifest)) {
    throw 'An experiment already exists; refusing to overwrite it.'
}
$existing = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match '\s-m\s+mcts\.422\.catalog_reproduction(?:\s|$)' }
if ($existing) {
    throw 'A catalog reproduction worker is already running.'
}

New-Item -ItemType Directory -Path $run -Force | Out-Null
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONHASHSEED = '0'
$stamp = [DateTimeOffset]::UtcNow.ToOffset([TimeSpan]::FromHours(8)).ToString('yyyyMMdd_HHmmss_fff')
$stdout = Join-Path $run "run_$stamp.log"
$stderr = Join-Path $run "stderr_$stamp.log"
$workerArguments = @('-u', '-m', 'mcts.422.catalog_reproduction', '--output-dir', $run)
if ($Resume) { $workerArguments += '--resume' }

# Use Win32_Process.Create to launch this script with an independent parent.
$worker = Start-Process -FilePath $python -ArgumentList $workerArguments `
    -WorkingDirectory (Join-Path $project 'src') -WindowStyle Hidden `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$record = [ordered]@{
    started_at = [DateTimeOffset]::UtcNow.ToOffset([TimeSpan]::FromHours(8)).ToString('o')
    launcher_pid = $PID
    python_launcher_pid = $worker.Id
    resume = [bool]$Resume
    output_dir = $run
    stdout_log = $stdout
    stderr_log = $stderr
    iterations_limit = $null
    wall_time_limit_seconds = $null
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run "launch_$stamp.json") -Encoding UTF8
$record | ConvertTo-Json
