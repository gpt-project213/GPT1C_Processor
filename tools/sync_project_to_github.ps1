# ============================================================
# sync_project_to_github.ps1
# Version: v6.0.0
# Date: 2026-03-07 (Asia/Almaty)
# Safe GitHub synchronizer for GPT1C_Processor
# ============================================================

param(
    [string]$ProjectRoot = "E:\GPT1C_Processor_analitic",
    [string]$Branch = "master",
    [string]$CommitMessage = "",
    [string]$RepoOwner = "gpt-project213",
    [string]$RepoName = "GPT1C_Processor",

    [switch]$RunSanitizer,
    [switch]$SkipPull,
    [switch]$SkipDependencyMap,
    [switch]$SkipSecretScan,
    [switch]$SkipHardcodedPathsLinter
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# ------------------------------------------------------------
# GLOBALS
# ------------------------------------------------------------

$script:ScriptPath = $MyInvocation.MyCommand.Path
$script:ScriptDir  = Split-Path -Parent $script:ScriptPath
$script:LogFile    = Join-Path $script:ScriptDir "sync_project_to_github.runtime.log.txt"

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

function Ensure-Dir {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Get-AlmatyNowString {
    try {
        $tz = [System.TimeZoneInfo]::FindSystemTimeZoneById("Central Asia Standard Time")
        $dt = [System.TimeZoneInfo]::ConvertTimeFromUtc([datetime]::UtcNow, $tz)
        return $dt.ToString("yyyy-MM-dd HH:mm:ss")
    }
    catch {
        return (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    }
}

function Write-Utf8NoBomFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Content
    )

    $dir = Split-Path -Parent $Path
    if ($dir) { Ensure-Dir $dir }

    $text = if ($Content -is [System.Array]) {
        ($Content -join [Environment]::NewLine)
    }
    else {
        [string]$Content
    }

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $text, $utf8NoBom)
}

function Append-Utf8NoBomLine {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Line
    )

    $dir = Split-Path -Parent $Path
    if ($dir) { Ensure-Dir $dir }

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
    try {
        $sw = New-Object System.IO.StreamWriter($fs, $utf8NoBom)
        try {
            $sw.WriteLine($Line)
        }
        finally {
            $sw.Dispose()
        }
    }
    finally {
        $fs.Dispose()
    }
}

function Log {
    param([Parameter(Mandatory = $true)][string]$Message)

    $line = "[{0}] {1}" -f (Get-AlmatyNowString), $Message
    Write-Host $line
    Append-Utf8NoBomLine -Path $script:LogFile -Line $line
}

function Fail {
    param([Parameter(Mandatory = $true)][string]$Message)

    Log "ERROR: $Message"
    exit 1
}

function Normalize-RelativePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $p = $Path.Trim()
    $p = $p -replace "\\", "/"

    if ($p.StartsWith("./")) {
        $p = $p.Substring(2)
    }

    if ($p.StartsWith('"') -and $p.EndsWith('"') -and $p.Length -ge 2) {
        $p = $p.Substring(1, $p.Length - 2)
    }

    return $p
}

function Exec-Git {
    param([Parameter(Mandatory = $true)][string[]]$Args)

    $cmdLine = "git " + ($Args -join " ")
    Log "CMD: $cmdLine"

    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        $output = @(& git @Args 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldEap
    }

    foreach ($line in $output) {
        Log ("OUT: " + [string]$line)
    }

    if ($exitCode -ne 0) {
        throw ("Git failed ({0}): {1}" -f $exitCode, $cmdLine)
    }

    return @($output | ForEach-Object { [string]$_ })
}

function Exec-External {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string[]]$Args
    )

    $cmdLine = $Exe + " " + ($Args -join " ")
    Log "CMD: $cmdLine"

    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        $output = @(& $Exe @Args 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldEap
    }

    foreach ($line in $output) {
        Log ("OUT: " + [string]$line)
    }

    return [pscustomobject]@{
        ExitCode = $exitCode
        Output   = @($output | ForEach-Object { [string]$_ })
    }
}

function Get-TrackedFiles {
    $files = @(Exec-Git @("-c", "core.quotepath=false", "ls-files"))
    $clean = New-Object System.Collections.Generic.List[string]

    foreach ($f in $files) {
        if ([string]::IsNullOrWhiteSpace($f)) { continue }
        $clean.Add((Normalize-RelativePath ([string]$f)))
    }

    return @($clean)
}

function Get-StagedFiles {
    $files = @(Exec-Git @("-c", "core.quotepath=false", "diff", "--cached", "--name-only"))
    $clean = New-Object System.Collections.Generic.List[string]

    foreach ($f in $files) {
        if ([string]::IsNullOrWhiteSpace($f)) { continue }
        $clean.Add((Normalize-RelativePath ([string]$f)))
    }

    return @($clean)
}

function Get-RepoRootRelative {
    param([Parameter(Mandatory = $true)][string]$AbsolutePath)

    $resolvedRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
    $resolvedFile = [System.IO.Path]::GetFullPath($AbsolutePath)

    if (-not $resolvedFile.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $null
    }

    $rel = $resolvedFile.Substring($resolvedRoot.Length).TrimStart('\','/')
    return (Normalize-RelativePath $rel)
}

function Git-Unstage-IfExists {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    try {
        & git reset --quiet -- $RelativePath 2>$null | Out-Null
    }
    catch {
    }
}

function Git-Unstage-ServiceFiles {
    $serviceFiles = @(
        (Get-RepoRootRelative $script:LogFile),
        "raw_links.txt",
        "repo_map.json",
        "AI_CONTEXT.md",
        "STATUS.md",
        "DEPENDENCY_MAP.md"
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Sort-Object -Unique

    foreach ($f in $serviceFiles) {
        Git-Unstage-IfExists -RelativePath $f
    }
}

# ------------------------------------------------------------
# DIAGNOSTICS
# ------------------------------------------------------------

function Assert-GitIgnoreContains {
    param([Parameter(Mandatory = $true)][string[]]$RequiredEntries)

    $gitignorePath = Join-Path $ProjectRoot ".gitignore"
    if (-not (Test-Path -LiteralPath $gitignorePath)) {
        Fail ".gitignore not found"
    }

    $content = Get-Content -LiteralPath $gitignorePath -ErrorAction Stop
    $missing = New-Object System.Collections.Generic.List[string]

    foreach ($entry in $RequiredEntries) {
        $found = $false
        foreach ($line in $content) {
            if ($line.Trim() -eq $entry.Trim()) {
                $found = $true
                break
            }
        }
        if (-not $found) {
            $missing.Add($entry)
        }
    }

    if ($missing.Count -gt 0) {
        foreach ($m in $missing) {
            Log "MISSING .gitignore ENTRY: $m"
        }
        Fail ".gitignore is missing required runtime-ignore rules"
    }

    Log ".gitignore runtime rules are present"
}

function Check-SensitiveTracked {
    $patterns = @(
        '^(?:\.env)$',
        '^config/(?:imap|roles|managers)\.json$',
        '^logs/',
        '^reports/',
        '^archive/',
        '^cache/',
        '^reports_state\.json$',
        '^ai_generation_state\.json$',
        '^ai_generation_queue\.json$',
        '^deletion_queue\.json$',
        '^daily_activity\.json$',
        '^tools/sync_project_to_github\.runtime\.log\.txt$'
    )

    $tracked = Get-TrackedFiles
    $bad = New-Object System.Collections.Generic.List[string]

    foreach ($f in $tracked) {
        foreach ($p in $patterns) {
            if ($f -match $p) {
                $bad.Add($f)
                break
            }
        }
    }

    if ($bad.Count -gt 0) {
        foreach ($x in ($bad | Sort-Object -Unique)) {
            Log "BAD TRACKED: $x"
        }
        Fail "Sensitive/runtime files are tracked by git"
    }

    Log "Sensitive/runtime tracked files: none"
}

function Test-PathLikelyBinary {
    param([Parameter(Mandatory = $true)][string]$Path)

    $ext = [System.IO.Path]::GetExtension($Path).ToLowerInvariant()
    $binaryExt = @(
        ".png",".jpg",".jpeg",".gif",".bmp",".webp",".ico",
        ".pdf",".zip",".7z",".rar",".xls",".xlsx",".xlsm",
        ".doc",".docx",".ppt",".pptx",".exe",".dll",".pyd",
        ".bin",".parquet",".feather",".sqlite",".db",".mp3",
        ".mp4",".avi",".mov",".wav"
    )

    return $binaryExt -contains $ext
}

function Scan-StagedFilesForSecrets {
    if ($SkipSecretScan) {
        Log "Secret scan skipped by switch"
        return
    }

    $staged = Get-StagedFiles
    if ($staged.Count -eq 0) {
        Log "No staged files for secret scan"
        return
    }

    $secretPatterns = @(
        '\b\d{8,12}:[A-Za-z0-9_-]{20,}\b',                                # Telegram token
        '\bsk-[A-Za-z0-9]{10,}\b',                                        # generic API key
        '(?i)\b(?:DEEPSEEK_API_KEY|OPENAI_API_KEY|TG_BOT_TOKEN|BOT_TOKEN|API_KEY|SECRET_KEY|PASSWORD|PASSWD|PWD|TOKEN)\s*[:=]\s*["'']?[^"''\s]+',
        '(?i)\bBearer\s+[A-Za-z0-9\._\-]{16,}\b',
        '(?i)\b(?:Authorization)\s*[:=]\s*["'']?Bearer\s+[A-Za-z0-9\._\-]{16,}\b'
    )

    $hits = New-Object System.Collections.Generic.List[string]

    foreach ($rel in $staged) {
        $full = Join-Path $ProjectRoot ($rel -replace '/', '\')
        if (-not (Test-Path -LiteralPath $full)) { continue }
        if (Test-PathLikelyBinary -Path $full) { continue }

        $text = ""
        try {
            $text = Get-Content -LiteralPath $full -Raw -ErrorAction Stop
        }
        catch {
            continue
        }

        foreach ($pattern in $secretPatterns) {
            $m = [regex]::Matches($text, $pattern)
            if ($m.Count -gt 0) {
                $hits.Add($rel)
                break
            }
        }
    }

    if ($hits.Count -gt 0) {
        foreach ($x in ($hits | Sort-Object -Unique)) {
            Log "SECRET SCAN HIT: $x"
        }
        Fail "Potential secret/token leak detected in staged files"
    }

    Log "Secret scan passed"
}

# ------------------------------------------------------------
# GENERATED FILES
# ------------------------------------------------------------

function Update-RawLinks {
    $base = "https://raw.githubusercontent.com/$RepoOwner/$RepoName/$Branch"
    $lines = @()
    $lines += "# generated_at: $(Get-AlmatyNowString)"
    $lines += "# timezone: Asia/Almaty"
    $lines += "# repository: https://github.com/$RepoOwner/$RepoName"
    $lines += "# branch: $Branch"
    $lines += ""

    foreach ($f in (Get-TrackedFiles | Sort-Object)) {
        $lines += "$base/$f"
    }

    Write-Utf8NoBomFile -Path (Join-Path $ProjectRoot "raw_links.txt") -Content $lines
    Log "Updated raw_links.txt"
}

function Update-RepoMap {
    $repoUrl = "https://github.com/$RepoOwner/$RepoName"
    $rawBase = "https://raw.githubusercontent.com/$RepoOwner/$RepoName/$Branch"

    $items = foreach ($f in (Get-TrackedFiles | Sort-Object)) {
        $name = Split-Path $f -Leaf
        $ext = [System.IO.Path]::GetExtension($name)
        $parts = $f.Split("/")
        $top = if ($parts.Count -gt 1) { $parts[0] } else { "." }
        $type = if ([string]::IsNullOrWhiteSpace($ext)) { "other" } else { $ext.TrimStart(".").ToLowerInvariant() }

        [pscustomobject]@{
            path       = $f
            name       = $name
            extension  = $ext
            type       = $type
            top_level  = $top
            github_url = "$repoUrl/blob/$Branch/$f"
            raw_url    = "$rawBase/$f"
        }
    }

    $sections = $items |
        Group-Object top_level |
        Sort-Object Name |
        ForEach-Object {
            [pscustomobject]@{
                section = $_.Name
                count   = $_.Count
                files   = ($_.Group | Sort-Object path)
            }
        }

    $result = [pscustomobject]@{
        generated_at = (Get-AlmatyNowString)
        timezone     = "Asia/Almaty"
        repo         = [pscustomobject]@{
            owner      = $RepoOwner
            name       = $RepoName
            branch     = $Branch
            github_url = $repoUrl
            raw_base   = $rawBase
        }
        summary      = [pscustomobject]@{
            total_files = @($items).Count
            top_levels  = @($sections).Count
        }
        sections     = $sections
    }

    $json = $result | ConvertTo-Json -Depth 12
    Write-Utf8NoBomFile -Path (Join-Path $ProjectRoot "repo_map.json") -Content $json
    Log "Updated repo_map.json"
}

function Update-AIContext {
    $repoUrl  = "https://github.com/$RepoOwner/$RepoName"
    $rawBase  = "https://raw.githubusercontent.com/$RepoOwner/$RepoName/$Branch"
    $tracked  = Get-TrackedFiles

    $parsers   = @($tracked | Where-Object { $_ -match '_parser\.py$|analyze_.*\.py$' } | Sort-Object)
    $reports   = @($tracked | Where-Object { $_ -match '(?:^|/)(?:.*report.*\.py)$' } | Sort-Object)
    $botFiles  = @($tracked | Where-Object { $_ -match '^bot/' } | Sort-Object)
    $templates = @($tracked | Where-Object { $_ -match '^templates/' } | Sort-Object)

    $lines = @()
    $lines += "# AI_CONTEXT.md"
    $lines += "<!-- AUTO-GENERATED. DO NOT EDIT MANUALLY -->"
    $lines += ""
    $lines += "Generated: $(Get-AlmatyNowString) (Asia/Almaty)"
    $lines += "Repository: $repoUrl"
    $lines += "RAW base: $rawBase"
    $lines += "Branch: $Branch"
    $lines += ""
    $lines += "## Purpose"
    $lines += "GPT1C_Processor / AI 1C PRO - processing dirty 1C Excel exports into analytical HTML/JSON/PDF reports and Telegram delivery."
    $lines += ""
    $lines += "## Reading order"
    $lines += "1. AI_CONTEXT.md"
    $lines += "2. STATUS.md"
    $lines += "3. DEPENDENCY_MAP.md"
    $lines += "4. repo_map.json"
    $lines += "5. raw_links.txt"
    $lines += ""

    $lines += "## Parsers"
    if ($parsers.Count -eq 0) { $lines += "- none" } else { foreach ($x in $parsers) { $lines += "- $x" } }
    $lines += ""

    $lines += "## Report scripts"
    if ($reports.Count -eq 0) { $lines += "- none" } else { foreach ($x in $reports) { $lines += "- $x" } }
    $lines += ""

    $lines += "## Bot files"
    if ($botFiles.Count -eq 0) { $lines += "- none" } else { foreach ($x in $botFiles) { $lines += "- $x" } }
    $lines += ""

    $lines += "## Templates"
    if ($templates.Count -eq 0) { $lines += "- none" } else { foreach ($x in $templates) { $lines += "- $x" } }

    Write-Utf8NoBomFile -Path (Join-Path $ProjectRoot "AI_CONTEXT.md") -Content $lines
    Log "Updated AI_CONTEXT.md"
}

function Update-Status {
    $repoUrl = "https://github.com/$RepoOwner/$RepoName"
    $rawBase = "https://raw.githubusercontent.com/$RepoOwner/$RepoName/$Branch"

    $trackedCount = @(Get-TrackedFiles).Count

    $checks = [ordered]@{
        "README.md"                  = (Test-Path (Join-Path $ProjectRoot "README.md"))
        "SECURITY.md"                = (Test-Path (Join-Path $ProjectRoot "SECURITY.md"))
        ".env.example"               = (Test-Path (Join-Path $ProjectRoot ".env.example"))
        "config/imap.example.json"   = (Test-Path (Join-Path $ProjectRoot "config\imap.example.json"))
        "config/managers.example.json" = (Test-Path (Join-Path $ProjectRoot "config\managers.example.json"))
        "config/roles.example.json"  = (Test-Path (Join-Path $ProjectRoot "config\roles.example.json"))
        "AI_CONTEXT.md"              = (Test-Path (Join-Path $ProjectRoot "AI_CONTEXT.md"))
        "DEPENDENCY_MAP.md"          = (Test-Path (Join-Path $ProjectRoot "DEPENDENCY_MAP.md"))
    }

    $lines = @()
    $lines += "# STATUS.md"
    $lines += "Generated: $(Get-AlmatyNowString) (Asia/Almaty)"
    $lines += "Repository: $repoUrl"
    $lines += "Branch: $Branch"
    $lines += "RAW base: $rawBase"
    $lines += ""
    $lines += "## Status"
    $lines += "- GitHub sync script is active"
    $lines += "- tracked files count: $trackedCount"
    $lines += ""
    $lines += "## Audit"
    foreach ($k in $checks.Keys) {
        $state = if ($checks[$k]) { "OK" } else { "MISSING" }
        $lines += "- $state :: $k"
    }

    Write-Utf8NoBomFile -Path (Join-Path $ProjectRoot "STATUS.md") -Content $lines
    Log "Updated STATUS.md"
}

function Update-DependencyMap {
    if ($SkipDependencyMap) {
        Log "Dependency map skipped by switch"
        return
    }

    $pyFiles = Get-ChildItem -Path $ProjectRoot -Recurse -File -Filter *.py -ErrorAction Stop |
        Where-Object {
            $_.FullName -notmatch '\\\.venv\\' -and
            $_.FullName -notmatch '\\venv\\' -and
            $_.FullName -notmatch '\\__pycache__\\' -and
            $_.FullName -notmatch '\\archive\\' -and
            $_.FullName -notmatch '\\cache\\' -and
            $_.FullName -notmatch '\\reports\\' -and
            $_.FullName -notmatch '\\logs\\'
        } |
        Sort-Object FullName

    $rows = New-Object System.Collections.Generic.List[object]

    foreach ($file in $pyFiles) {
        $rel = Get-RepoRootRelative -AbsolutePath $file.FullName
        if (-not $rel) { continue }

        $text = ""
        try {
            $text = Get-Content -LiteralPath $file.FullName -Raw -ErrorAction Stop
        }
        catch {
            continue
        }

        $imports = New-Object System.Collections.Generic.List[string]

        $rx1 = [regex]'(?m)^\s*import\s+([A-Za-z0-9_\. ,]+)'
        foreach ($m in $rx1.Matches($text)) {
            $raw = $m.Groups[1].Value.Split(",")
            foreach ($item in $raw) {
                $name = $item.Trim()
                if ($name -match '\s+as\s+') {
                    $name = ($name -split '\s+as\s+')[0].Trim()
                }
                if (-not [string]::IsNullOrWhiteSpace($name)) {
                    $imports.Add($name)
                }
            }
        }

        $rx2 = [regex]'(?m)^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+'
        foreach ($m in $rx2.Matches($text)) {
            $name = $m.Groups[1].Value.Trim()
            if (-not [string]::IsNullOrWhiteSpace($name)) {
                $imports.Add($name)
            }
        }

        $rows.Add([pscustomobject]@{
            file    = $rel
            imports = @($imports | Sort-Object -Unique)
        })
    }

    $lines = @()
    $lines += "# DEPENDENCY_MAP.md"
    $lines += "Generated: $(Get-AlmatyNowString) (Asia/Almaty)"
    $lines += ""
    $lines += "## Python import map"
    $lines += ""

    foreach ($row in ($rows | Sort-Object file)) {
        $lines += "### $($row.file)"
        if ($row.imports.Count -eq 0) {
            $lines += "- no imports detected"
        }
        else {
            foreach ($imp in $row.imports) {
                $lines += "- $imp"
            }
        }
        $lines += ""
    }

    Write-Utf8NoBomFile -Path (Join-Path $ProjectRoot "DEPENDENCY_MAP.md") -Content $lines
    Log "Updated DEPENDENCY_MAP.md"
}

# ------------------------------------------------------------
# OPTIONAL TASKS
# ------------------------------------------------------------

function Run-HardcodedPathsLinter {
    if ($SkipHardcodedPathsLinter) {
        Log "Hardcoded-paths linter skipped by switch"
        return
    }

    $lintScript = Join-Path $ProjectRoot "check_hardcoded_paths.py"
    if (-not (Test-Path -LiteralPath $lintScript)) {
        Log "Hardcoded-paths linter not found, skip"
        return
    }

    $pythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        $pythonExe = "python"
    }

    Log "Running hardcoded-paths linter"
    $result = Exec-External -Exe $pythonExe -Args @("-X", "utf8", $lintScript)

    if ($result.ExitCode -ne 0) {
        Fail "check_hardcoded_paths.py failed"
    }

    Log "Hardcoded-paths linter completed"
}

function Run-OptionalSanitizer {
    if (-not $RunSanitizer) {
        Log "Sanitizer skipped"
        return
    }

    $sanitizer = Join-Path $script:ScriptDir "sanitize_logs_for_github.ps1"
    if (-not (Test-Path -LiteralPath $sanitizer)) {
        Log "sanitize_logs_for_github.ps1 not found, skip"
        return
    }

    Log "Running sanitize_logs_for_github.ps1"
    $result = Exec-External -Exe "powershell" -Args @(
        "-ExecutionPolicy", "Bypass",
        "-File", $sanitizer,
        "-ProjectRoot", $ProjectRoot
    )

    if ($result.ExitCode -ne 0) {
        Fail "sanitize_logs_for_github.ps1 failed"
    }

    Log "Sanitizer completed"
}

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

try {
    if (-not (Test-Path -LiteralPath $ProjectRoot)) {
        Fail "ProjectRoot not found: $ProjectRoot"
    }

    Ensure-Dir $script:ScriptDir

    Log "START GITHUB SYNC"
    Log "ProjectRoot = $ProjectRoot"
    Log "TargetBranch = $Branch"
    Log "Repo = $RepoOwner/$RepoName"
    Log "RuntimeLogFile = $script:LogFile"

    Set-Location $ProjectRoot

    if (-not (Test-Path -LiteralPath ".git")) {
        Fail "This directory is not a git repository"
    }

    $currentDir = (Get-Location).Path
    Log "CurrentDirectory = $currentDir"

    $currentBranch = ((Exec-Git @("rev-parse", "--abbrev-ref", "HEAD")) -join "`n").Trim()
    if ([string]::IsNullOrWhiteSpace($currentBranch)) {
        Fail "Cannot detect current branch"
    }

    Log "CurrentBranch = $currentBranch"

    if ($currentBranch -ne $Branch) {
        Fail "Wrong branch. Expected: $Branch; current: $currentBranch"
    }

    $origin = ((Exec-Git @("remote", "get-url", "origin")) -join "`n").Trim()
    if ([string]::IsNullOrWhiteSpace($origin)) {
        Fail "Remote origin not found"
    }
    Log "Origin = $origin"

    Assert-GitIgnoreContains -RequiredEntries @(
        ".env",
        "logs/",
        "reports/",
        "cache/",
        "archive/",
        "tools/sync_project_to_github.runtime.log.txt"
    )

    Check-SensitiveTracked

    Log "Git status before pre-pull phase"
    $statusBefore = @(Exec-Git @("status", "--short"))
    if ($statusBefore.Count -eq 0) {
        Log "Working tree is clean before sync"
    }
    else {
        foreach ($line in $statusBefore) {
            Log "STATUS: $line"
        }
    }

    # --------------------------------------------------------
    # PRE-PULL AUTO-COMMIT
    # --------------------------------------------------------

    $porcelain = @(Exec-Git @("status", "--porcelain"))
    if ($porcelain.Count -gt 0) {
        Log "Local changes detected before pull"

        Exec-Git @("add", ".") | Out-Null
        Git-Unstage-ServiceFiles

        $preStaged = Get-StagedFiles
        if ($preStaged.Count -gt 0) {
            foreach ($f in $preStaged) {
                Log "PRE-PULL STAGED: $f"
            }

            Scan-StagedFilesForSecrets

            $preMsg = "auto commit before pull $(Get-AlmatyNowString)"
            Log "PrePullCommitMessage = $preMsg"
            Exec-Git @("commit", "-m", $preMsg) | Out-Null
            Log "Pre-pull auto-commit created"
        }
        else {
            Log "Only runtime/service files changed before pull; no pre-pull commit needed"
        }
    }
    else {
        Log "No local changes before pull"
    }

    # --------------------------------------------------------
    # PULL
    # --------------------------------------------------------

    if (-not $SkipPull) {
        Log "Running git pull --rebase"
        try {
            Exec-Git @("pull", "--rebase", "origin", $Branch) | Out-Null
            Log "Pull --rebase completed"
        }
        catch {
            Log "REBASE FAILURE DETECTED"
            Log "MANUAL ACTION REQUIRED: resolve conflicts, then run:"
            Log "  git rebase --continue"
            Log "or abort:"
            Log "  git rebase --abort"
            throw
        }
    }
    else {
        Log "Pull skipped by switch"
    }

    # --------------------------------------------------------
    # GENERATED META
    # --------------------------------------------------------

    Update-RawLinks
    Update-RepoMap
    Update-AIContext
    Update-Status
    Update-DependencyMap
    Run-HardcodedPathsLinter
    Run-OptionalSanitizer

    # --------------------------------------------------------
    # FINAL ADD / COMMIT / PUSH
    # --------------------------------------------------------

    Log "Staging changes with git add ."
    Exec-Git @("add", ".") | Out-Null
    Git-Unstage-ServiceFiles
    Log "git add completed"

    $staged = Get-StagedFiles
    if ($staged.Count -eq 0) {
        Log "No staged changes after update. Nothing to commit or push"
        Log "DONE"
        exit 0
    }

    foreach ($f in $staged) {
        Log "FINAL STAGED: $f"
    }

    Scan-StagedFilesForSecrets

    if ([string]::IsNullOrWhiteSpace($CommitMessage)) {
        if ($staged.Count -le 8) {
            $CommitMessage = "sync: " + ($staged -join ", ")
        }
        else {
            $CommitMessage = "sync project updates ($($staged.Count) files)"
        }
    }

    Log "CommitMessage = $CommitMessage"

    Exec-Git @("commit", "-m", $CommitMessage) | Out-Null
    Log "Commit created"

    Exec-Git @("push", "origin", $Branch) | Out-Null
    Log "Push completed"

    $lastCommit = ((Exec-Git @("log", "-1", "--oneline")) -join "`n").Trim()
    Log "LastCommit = $lastCommit"

    Log "SYNC SUMMARY"
    Log "SyncedFilesCount = $($staged.Count)"
    foreach ($f in $staged) {
        Log "Synced: $f"
    }

    Log "DONE"
    exit 0
}
catch {
    $msg = $_.Exception.Message
    if ([string]::IsNullOrWhiteSpace($msg)) {
        $msg = [string]$_
    }

    Log "FATAL: $msg"
    exit 1
}
