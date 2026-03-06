# tools/sync_project_to_github.ps1
# v2.0.1 · 2026-03-06 (Asia/Almaty)

[CmdletBinding()]
param(
    [string]$ProjectRoot = "E:\GPT1C_Processor_analitic",
    [string]$Branch = "master",
    [string]$CommitMessage = ""
)

$ErrorActionPreference = "Stop"

function Stop-WithError {
    param([string]$Message)

    Write-Host ""
    Write-Host ("ERROR: " + $Message) -ForegroundColor Red

    if ($script:LogFile) {
        Add-Content -Path $script:LogFile -Value ("[{0}] ERROR: {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message) -Encoding UTF8
    }

    exit 1
}

function Write-Log {
    param(
        [string]$Message,
        [string]$Color = "Gray"
    )

    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line -ForegroundColor $Color

    if ($script:LogFile) {
        Add-Content -Path $script:LogFile -Value $line -Encoding UTF8
    }
}

function Run-Git {
    param(
        [Parameter(Mandatory = $true)][string[]]$Args,
        [switch]$AllowNonZero
    )

    $gitExe = (Get-Command git -ErrorAction Stop).Source
    $argLine = ($Args | ForEach-Object {
        if ($_ -match '\s|["]') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    }) -join ' '

    $stdoutFile = [System.IO.Path]::GetTempFileName()
    $stderrFile = [System.IO.Path]::GetTempFileName()

    try {
        $proc = Start-Process `
            -FilePath $gitExe `
            -ArgumentList $argLine `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutFile `
            -RedirectStandardError $stderrFile

        $stdout = @()
        $stderr = @()

        if (Test-Path $stdoutFile) {
            $stdout = Get-Content -Path $stdoutFile -ErrorAction SilentlyContinue
        }

        if (Test-Path $stderrFile) {
            $stderr = Get-Content -Path $stderrFile -ErrorAction SilentlyContinue
        }

        Add-Content -Path $script:LogFile -Value ("[{0}] CMD: git {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), ($Args -join " ")) -Encoding UTF8

        foreach ($line in $stdout) {
            Add-Content -Path $script:LogFile -Value ("    OUT: " + $line) -Encoding UTF8
        }

        foreach ($line in $stderr) {
            Add-Content -Path $script:LogFile -Value ("    ERR: " + $line) -Encoding UTF8
        }

        if (-not $AllowNonZero -and $proc.ExitCode -ne 0) {
            $message = "git " + ($Args -join " ") + " failed with exit code " + $proc.ExitCode
            if ($stderr.Count -gt 0) {
                $message += ". stderr: " + ($stderr -join " | ")
            }
            Stop-WithError $message
        }

        return [pscustomobject]@{
            ExitCode = $proc.ExitCode
            StdOut   = $stdout
            StdErr   = $stderr
            All      = @($stdout + $stderr)
        }
    }
    finally {
        Remove-Item -Path $stdoutFile -ErrorAction SilentlyContinue -Force
        Remove-Item -Path $stderrFile -ErrorAction SilentlyContinue -Force
    }
}

if (-not (Test-Path $ProjectRoot)) {
    Write-Host "ERROR: Project root not found: $ProjectRoot" -ForegroundColor Red
    exit 1
}

$logsDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$script:LogFile = Join-Path $logsDir ("github_sync_{0}.log" -f $timestamp)

Write-Log "START GITHUB SYNC" "Cyan"
Write-Log ("ProjectRoot = " + $ProjectRoot) "Cyan"
Write-Log ("TargetBranch = " + $Branch) "Cyan"

Set-Location $ProjectRoot
Write-Log ("CurrentDirectory = " + (Get-Location).Path) "Green"

if (-not (Test-Path ".git")) {
    Stop-WithError "No .git directory found. This is not a git repository."
}

$repoRoot = ((Run-Git -Args @("rev-parse", "--show-toplevel")).StdOut -join "`n").Trim()
if (-not $repoRoot) {
    Stop-WithError "Cannot detect repository root."
}
Write-Log ("RepoRoot = " + $repoRoot) "Green"

$origin = ((Run-Git -Args @("remote", "get-url", "origin")).StdOut -join "`n").Trim()
if (-not $origin) {
    Stop-WithError "Remote origin not found."
}
Write-Log ("Origin = " + $origin) "Green"

$currentBranch = ((Run-Git -Args @("rev-parse", "--abbrev-ref", "HEAD")).StdOut -join "`n").Trim()
Write-Log ("CurrentBranch = " + $currentBranch) "Green"

if ($currentBranch -ne $Branch) {
    Stop-WithError ("Current branch '" + $currentBranch + "' does not match target branch '" + $Branch + "'.")
}

Write-Log "Checking tracked sensitive/runtime files..." "Cyan"

$patterns = @(
    '^\.(env)$',
    '^config/(imap|roles|managers)\.json$',
    '^logs/',
    '^reports/',
    '^archive/',
    '^cache/',
    '^reports_state\.json$',
    '^ai_generation_state\.json$',
    '^ai_generation_queue\.json$',
    '^deletion_queue\.json$'
)

$trackedFiles = @((Run-Git -Args @("ls-files")).StdOut)
$badTracked = New-Object System.Collections.Generic.List[string]

foreach ($file in $trackedFiles) {
    foreach ($pattern in $patterns) {
        if ($file -match $pattern) {
            $badTracked.Add($file)
            break
        }
    }
}

if ($badTracked.Count -gt 0) {
    Write-Log "Tracked sensitive/runtime files found:" "Yellow"
    $badTracked | Sort-Object -Unique | ForEach-Object { Write-Log ("  " + $_) "Yellow" }
    Stop-WithError "Sensitive or runtime files are tracked by git. Sync stopped."
}

Write-Log "Tracked sensitive/runtime files: none" "Green"

Write-Log "Git status before sync:" "Cyan"
$statusBefore = @((Run-Git -Args @("status", "--short")).StdOut)
if ($statusBefore.Count -eq 0) {
    Write-Log "Working tree is clean before sync." "Green"
} else {
    foreach ($line in $statusBefore) {
        Write-Log ("  " + $line) "Gray"
    }
}

Write-Log "Running git pull --rebase..." "Cyan"
$pullResult = Run-Git -Args @("pull", "--rebase", "origin", $Branch)
foreach ($line in $pullResult.StdOut) {
    Write-Log ("  " + $line) "Gray"
}
foreach ($line in $pullResult.StdErr) {
    Write-Log ("  " + $line) "DarkGray"
}
Write-Log "Pull --rebase completed." "Green"

Write-Log "Staging changes with git add ." "Cyan"
Run-Git -Args @("add", ".") | Out-Null
Write-Log "git add completed." "Green"

$stagedFiles = @((Run-Git -Args @("diff", "--cached", "--name-only")).StdOut)
if ($stagedFiles.Count -eq 0) {
    Write-Log "No staged changes. Nothing to commit or push." "Yellow"
    Write-Log ("LogFile = " + $script:LogFile) "Cyan"
    exit 0
}

Write-Log "Staged files:" "Cyan"
foreach ($file in $stagedFiles) {
    Write-Log ("  " + $file) "Gray"
}

if ([string]::IsNullOrWhiteSpace($CommitMessage)) {
    if ($stagedFiles.Count -le 8) {
        $shortList = ($stagedFiles -join ", ")
        $CommitMessage = "sync: $shortList"
    } else {
        $CommitMessage = "sync project updates ($($stagedFiles.Count) files)"
    }
}

Write-Log ("CommitMessage = " + $CommitMessage) "Cyan"

Write-Log "Creating commit..." "Cyan"
$commitResult = Run-Git -Args @("commit", "-m", $CommitMessage)
foreach ($line in $commitResult.StdOut) {
    Write-Log ("  " + $line) "Gray"
}
foreach ($line in $commitResult.StdErr) {
    Write-Log ("  " + $line) "DarkGray"
}
Write-Log "Commit created." "Green"

Write-Log "Pushing to GitHub..." "Cyan"
$pushResult = Run-Git -Args @("push", "origin", $Branch)
foreach ($line in $pushResult.StdOut) {
    Write-Log ("  " + $line) "Gray"
}
foreach ($line in $pushResult.StdErr) {
    Write-Log ("  " + $line) "DarkGray"
}
Write-Log "Push completed." "Green"

$lastCommit = (((Run-Git -Args @("log", "-1", "--oneline")).StdOut) -join "`n").Trim()
Write-Log ("LastCommit = " + $lastCommit) "Green"

Write-Log "SYNC SUMMARY" "Cyan"
Write-Log ("SyncedFilesCount = " + $stagedFiles.Count) "Green"
foreach ($file in $stagedFiles) {
    Write-Log ("Synced: " + $file) "Green"
}

Write-Log ("LogFile = " + $script:LogFile) "Cyan"
Write-Log "DONE" "Cyan"