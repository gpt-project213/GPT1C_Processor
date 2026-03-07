# RUN:
# powershell -ExecutionPolicy Bypass -File "E:\GPT1C_Processor_analitic\tools\sync_project_to_github.ps1"
#
# RUN WITH MESSAGE:
# powershell -ExecutionPolicy Bypass -File "E:\GPT1C_Processor_analitic\tools\sync_project_to_github.ps1" -CommitMessage "update project status"

param(
    [string]$ProjectRoot = "E:\GPT1C_Processor_analitic",
    [string]$Branch = "master",
    [string]$CommitMessage = "",
    [string]$RepoOwner = "gpt-project213",
    [string]$RepoName = "GPT1C_Processor"
)

$ErrorActionPreference = "Stop"

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    if ($script:LogFile) {
        Add-Content -Path $script:LogFile -Value $line -Encoding UTF8
    }
}

function Fail {
    param([string]$Message)
    Log ("ERROR: " + $Message)
    exit 1
}

function Set-Utf8 {
    param(
        [string]$Path,
        [string[]]$Lines
    )
    $dir = Split-Path -Parent $Path
    if ($dir) { Ensure-Dir $dir }
    Set-Content -Path $Path -Value $Lines -Encoding UTF8
}

function Exec-Git {
    param([string[]]$GitArgs)

    if (-not $GitArgs -or $GitArgs.Count -eq 0) {
        throw "Exec-Git called with empty args"
    }

    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        $output = @(& git @GitArgs 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldEap
    }

    if ($script:LogFile) {
        Add-Content -Path $script:LogFile -Value ("    CMD: git " + ($GitArgs -join " ")) -Encoding UTF8
        foreach ($x in $output) {
            Add-Content -Path $script:LogFile -Value ("    OUT: " + [string]$x) -Encoding UTF8
        }
    }

    if ($exitCode -ne 0) {
        throw ("git " + ($GitArgs -join " ") + " failed: " + (($output | ForEach-Object { [string]$_ }) -join " | "))
    }

    return @($output | ForEach-Object { [string]$_ })
}

function Get-TrackedFiles {
    $files = @(Exec-Git @("-c", "core.quotepath=false", "ls-files"))
    $clean = @()

    foreach ($f in $files) {
        if ([string]::IsNullOrWhiteSpace($f)) { continue }

        $s = [string]$f
        $s = $s.Trim()

        if ($s.StartsWith('"') -and $s.EndsWith('"') -and $s.Length -ge 2) {
            $s = $s.Substring(1, $s.Length - 2)
        }

        $clean += $s
    }

    return $clean
}

function Check-SensitiveTracked {
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

    $bad = New-Object System.Collections.Generic.List[string]
    $tracked = Get-TrackedFiles

    foreach ($f in $tracked) {
        foreach ($p in $patterns) {
            if ($f -match $p) {
                $bad.Add($f)
                break
            }
        }
    }

    if ($bad.Count -gt 0) {
        foreach ($x in ($bad | Sort-Object -Unique)) { Log ("BAD TRACKED: " + $x) }
        Fail "Sensitive or runtime files are tracked by git"
    }

    Log "Sensitive/runtime tracked files: none"
}

function Update-RawLinks {
    $baseRaw = "https://raw.githubusercontent.com/$RepoOwner/$RepoName/$Branch"
    $lines = @()
    $lines += "# RAW links generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    $lines += "# Repository: https://github.com/$RepoOwner/$RepoName"
    $lines += "# Branch: $Branch"
    $lines += ""

    foreach ($f in (Get-TrackedFiles | Sort-Object)) {
        $path = ([string]$f).Trim() -replace "\\", "/"
        if ($path.StartsWith('"') -and $path.EndsWith('"') -and $path.Length -ge 2) {
            $path = $path.Substring(1, $path.Length - 2)
        }
        $lines += "$baseRaw/$path"
    }

    Set-Utf8 -Path (Join-Path $ProjectRoot "raw_links.txt") -Lines $lines
    Log "Updated raw_links.txt"
}

function Update-RepoMap {
    $repoUrl = "https://github.com/$RepoOwner/$RepoName"
    $rawBase = "https://raw.githubusercontent.com/$RepoOwner/$RepoName/$Branch"
    $items = @()

    foreach ($f in (Get-TrackedFiles | Sort-Object)) {
        $path = ([string]$f).Trim() -replace "\\", "/"

        if ($path.StartsWith('"') -and $path.EndsWith('"') -and $path.Length -ge 2) {
            $path = $path.Substring(1, $path.Length - 2)
        }

        $parts = $path.Split("/")
        $name = $parts[$parts.Length - 1]

        $ext = ""
        if ($name -match '\.') {
            $dotIndex = $name.LastIndexOf(".")
            if ($dotIndex -ge 0) {
                $ext = $name.Substring($dotIndex)
            }
        }

        $top = if ($path -match "/") { $parts[0] } else { "." }
        $type = if ([string]::IsNullOrWhiteSpace($ext)) { "other" } else { $ext.TrimStart(".").ToLower() }

        $items += [pscustomobject]@{
            path       = $path
            name       = $name
            extension  = $ext
            type       = $type
            top_level  = $top
            github_url = "$repoUrl/blob/$Branch/$path"
            raw_url    = "$rawBase/$path"
        }
    }

    $sections = $items | Group-Object top_level | Sort-Object Name | ForEach-Object {
        [pscustomobject]@{
            section = $_.Name
            count   = $_.Count
            files   = ($_.Group | Sort-Object path)
        }
    }

    $result = [pscustomobject]@{
        generated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
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

    $json = $result | ConvertTo-Json -Depth 10
    Set-Content -Path (Join-Path $ProjectRoot "repo_map.json") -Value $json -Encoding UTF8
    Log "Updated repo_map.json"
}

function Update-Status {
    $repoUrl = "https://github.com/$RepoOwner/$RepoName"
    $rawBase = "https://raw.githubusercontent.com/$RepoOwner/$RepoName/$Branch"

    $hasReadme          = Test-Path (Join-Path $ProjectRoot "README.md")
    $hasSecurity        = Test-Path (Join-Path $ProjectRoot "SECURITY.md")
    $hasEnvExample      = Test-Path (Join-Path $ProjectRoot ".env.example")
    $hasImapExample     = Test-Path (Join-Path $ProjectRoot "config\imap.example.json")
    $hasManagersExample = Test-Path (Join-Path $ProjectRoot "config\managers.example.json")
    $hasRolesExample    = Test-Path (Join-Path $ProjectRoot "config\roles.example.json")
    $trackedCount       = @(Get-TrackedFiles).Count

    $gapLines = @()
    if (-not $hasReadme)           { $gapLines += "- README.md" }
    if (-not $hasSecurity)         { $gapLines += "- SECURITY.md" }
    if (-not $hasEnvExample)       { $gapLines += "- .env.example" }
    if (-not $hasImapExample)      { $gapLines += "- config/imap.example.json" }
    if (-not $hasManagersExample)  { $gapLines += "- config/managers.example.json" }
    if (-not $hasRolesExample)     { $gapLines += "- config/roles.example.json" }
    if ($gapLines.Count -eq 0)    { $gapLines += "- base audit layer is complete" }

    $lines = @()
    $lines += "# STATUS.md"
    $lines += "Version: v$(Get-Date -Format 'yyyy-MM-dd')"
    $lines += "Timezone: Asia/Almaty"
    $lines += "Status: auto-updated by sync_project_to_github.ps1"
    $lines += ""
    $lines += "# GPT1C_Processor / AI 1C PRO - STATUS"
    $lines += ""
    $lines += "## 1. Project"
    $lines += ""
    $lines += "Local production project: E:\GPT1C_Processor_analitic"
    $lines += ""
    $lines += "GitHub repository: $repoUrl"
    $lines += ""
    $lines += "Branch: $Branch"
    $lines += ""
    $lines += "RAW base: $rawBase"
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## 2. Confirmed"
    $lines += ""
    $lines += "- production code is published to GitHub"
    $lines += "- local GitHub sync works"
    $lines += "- raw_links.txt generation is built in"
    $lines += "- repo_map.json generation is built in"
    $lines += "- STATUS.md generation is built in"
    $lines += "- AI_CONTEXT.md generation is built in"
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## 3. Tracked files"
    $lines += ""
    $lines += "- tracked files count: $trackedCount"
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## 4. Current priority"
    $lines += ""
    $lines += "- sync all production .py files to GitHub"
    $lines += "- fix inventory_summary.py pattern mismatch"
    $lines += "- fix dso_aging_report.py (synthetic aging, hardcoded /30)"
    $lines += "- fix revenue_concentration_report.py (normalize_client_name)"
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## 5. Audit layer gaps"
    $lines += ""
    $lines += $gapLines
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## 6. Purpose"
    $lines += ""
    $lines += "- short project status"
    $lines += "- recovery point"
    $lines += "- GitHub audit checkpoint"
    $lines += "- current audit layer reference"
    $lines += ""

    Set-Utf8 -Path (Join-Path $ProjectRoot "STATUS.md") -Lines $lines
    Log "Updated STATUS.md"
}

function Update-AiContext {
    $repoUrl  = "https://github.com/$RepoOwner/$RepoName"
    $rawBase  = "https://raw.githubusercontent.com/$RepoOwner/$RepoName/$Branch"
    $syncedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    $tracked   = Get-TrackedFiles
    $parsers   = @($tracked | Where-Object { $_ -match "_parser\.py$" } | Sort-Object)
    $botFiles  = @($tracked | Where-Object { $_ -match "^bot/" } | Sort-Object)
    $templates = @($tracked | Where-Object { $_ -match "^templates/" } | Sort-Object)
    $prompts   = @($tracked | Where-Object { $_ -match "\.txt$" } | Sort-Object)

    $lines = @()
    $lines += "# AI_CONTEXT.md"
    $lines += "<!-- AUTO-GENERATED by sync_project_to_github.ps1 - DO NOT EDIT MANUALLY -->"
    $lines += "<!-- Updated: $syncedAt (Asia/Almaty) -->"
    $lines += ""
    $lines += "## Project"
    $lines += ""
    $lines += "**GPT1C_Processor / AI 1C PRO**"
    $lines += ""
    $lines += "Automated pipeline for processing 1C Excel exports (debt, sales, gross profit, inventory, expenses),"
    $lines += "generating HTML/JSON/PDF analytics reports, and delivering them via Telegram bot with AI analysis."
    $lines += ""
    $lines += "- Repository: $repoUrl"
    $lines += "- RAW base: $rawBase"
    $lines += "- Branch: $Branch"
    $lines += "- Timezone: Asia/Almaty"
    $lines += "- Language: Python 3.11+"
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## How AI Should Read This Project"
    $lines += ""
    $lines += "Read files in this exact order to minimize wasted context:"
    $lines += ""
    $lines += "1. AI_CONTEXT.md (this file) - project map"
    $lines += "2. STATUS.md - current state and priorities"
    $lines += "3. DEPENDENCY_MAP.md - full import graph and data flow (9 layers)"
    $lines += "4. repo_map.json - machine-readable file index"
    $lines += "5. raw_links.txt - direct RAW URLs to all files"
    $lines += "6. Code: start with core layer, then parsers, then reports, then bot"
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## Architecture (9 Layers)"
    $lines += ""
    $lines += "Layer 0: utils_common.py, send_tg.py, cleanup_cache.py, tools/"
    $lines += "  Role: Utilities with no project imports"
    $lines += ""
    $lines += "Layer 1: config.py"
    $lines += "  Role: Paths, TZ, logging, managers config"
    $lines += ""
    $lines += "Layer 2: utils_excel.py, utils.py"
    $lines += "  Role: Excel cleaning, Jinja env, formatters"
    $lines += ""
    $lines += "Layer 3: *_parser.py"
    $lines += "  Role: xlsx to DataFrame/dict"
    $lines += ""
    $lines += "Layer 4: debt/sales/gross/inventory/expenses _report.py"
    $lines += "  Role: Excel to HTML and JSON"
    $lines += ""
    $lines += "Layer 5: dso/rfm/concentration/turnover/net_profit/profitability"
    $lines += "  Role: JSON to analytics HTML (safe to modify, no project imports)"
    $lines += ""
    $lines += "Layer 6: ai_analyzer.py"
    $lines += "  Role: HTML to AI txt/html via DeepSeek or OpenAI"
    $lines += ""
    $lines += "Layer 7: imap_fetcher.py, run_pipeline*.py"
    $lines += "  Role: Orchestrators"
    $lines += ""
    $lines += "Layer 8: bot/send_reports.py + bot/*.py"
    $lines += "  Role: Telegram UI"
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## Entry Points"
    $lines += ""
    $lines += "bot/send_reports.py       - Main bot, scheduler, menu, delivery"
    $lines += "run_pipeline_all_mp.py    - Multi-process pipeline (all report types)"
    $lines += "run_pipeline.py           - Simplified pipeline (debt only)"
    $lines += "imap_fetcher.py           - IMAP email to reports/queue/"
    $lines += "run_new_reports_now.py    - Manual trigger"
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## Parsers (Layer 3)"
    $lines += ""
    foreach ($f in $parsers) { $lines += "- $f" }
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## Core Report Generators (Layer 4)"
    $lines += ""
    $lines += "debt_auto_report.py     : xlsx -> reports/html/debt_*.html + reports/json/debt_*.json"
    $lines += "sales_report.py         : xlsx -> reports/html/sales_*.html"
    $lines += "gross_report.py         : xlsx -> reports/html/gross_*.html"
    $lines += "gross_report_pct.py     : xlsx -> reports/html/gross_pct_*.html"
    $lines += "inventory.py            : xlsx -> reports/html/inventory_*.html + reports/json/"
    $lines += "expenses_report.py      : xlsx -> reports/html/expenses_*.html + reports/json/"
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## Analytics (Layer 5) - reads JSON, writes reports/analytics/"
    $lines += ""
    $lines += "dso_aging_report.py              : reads debt + sales JSON"
    $lines += "rfm_clients_report.py            : reads sales JSON"
    $lines += "revenue_concentration_report.py  : reads sales JSON"
    $lines += "inventory_turnover_report.py     : reads inventory + sales JSON"
    $lines += "net_profit_report.py             : reads gross + expenses JSON"
    $lines += "sales_profitability_report.py    : reads sales + gross JSON"
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## Bot Modules (Layer 8)"
    $lines += ""
    foreach ($f in $botFiles) { $lines += "- $f" }
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## HTML Templates"
    $lines += ""
    foreach ($f in $templates) { $lines += "- $f" }
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## AI Prompts"
    $lines += ""
    foreach ($f in $prompts) { $lines += "- $f" }
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## Data Flow (summary)"
    $lines += ""
    $lines += "Email (IMAP)"
    $lines += "  imap_fetcher.py -> reports/queue/"
    $lines += "    run_pipeline_all_mp.py routes by filename:"
    $lines += "      DEBT    -> debt_auto_report.py  -> html + json/debt_ext_*.json"
    $lines += "      SALES   -> sales_report.py      -> html + json/sales_*.json"
    $lines += "      GROSS   -> gross_report.py      -> html + json/gross_*.json"
    $lines += "      INVENT  -> inventory.py         -> html + json/inventory_*.json"
    $lines += "      EXPENSE -> expenses_report.py   -> html + json/expenses_*.json"
    $lines += ""
    $lines += "reports/json/ -> Analytics layer (dso, rfm, concentration, turnover, net_profit)"
    $lines += "             -> reports/analytics/"
    $lines += ""
    $lines += "reports/html/ -> ai_analyzer.py -> reports/ai/ (txt + html)"
    $lines += ""
    $lines += "bot/send_reports.py (APScheduler):"
    $lines += "  every 10 min : run_pipeline_all_mp.py"
    $lines += "  09:00        : inventory summary -> admin"
    $lines += "  14:00        : silence_alerts + opportunity_loss -> managers"
    $lines += "  20:00        : gross report"
    $lines += "  21:00        : sales + silence_alerts"
    $lines += "  22:00        : daily_analytics (all 6 analytics)"
    $lines += "  23:00        : daily_summary -> admin"
    $lines += "  Mon 10:00    : weekly_analytics"
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## Config Files"
    $lines += ""
    $lines += ".env                       : TG_BOT_TOKEN, DEEPSEEK_API_KEY, TZ, paths"
    $lines += "config/managers.json       : Manager names and chat_id"
    $lines += "config/roles.json          : Admin, subadmin_scopes"
    $lines += "config/imap.json           : IMAP server credentials"
    $lines += "config/pattern_config.yaml : Regex for report type detection"
    $lines += ""
    $lines += "---"
    $lines += ""
    $lines += "## Roles"
    $lines += ""
    $lines += "admin    : Vadim (7422963573)     - sees everything"
    $lines += "subadmin : Alena (188939016)      - sees Magira + Oksana reports"
    $lines += "manager  : Oksana, Magira, Ergali - sees own reports only"
    $lines += ""

    Set-Utf8 -Path (Join-Path $ProjectRoot "AI_CONTEXT.md") -Lines $lines
    Log "Updated AI_CONTEXT.md"
}

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if (-not (Test-Path $ProjectRoot)) {
    Write-Host ("ERROR: ProjectRoot not found: " + $ProjectRoot)
    exit 1
}

$logsDir = Join-Path $ProjectRoot "logs"
Ensure-Dir $logsDir
$script:LogFile = Join-Path $logsDir ("github_sync_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

Log "START GITHUB SYNC"
Log ("ProjectRoot = " + $ProjectRoot)
Log ("TargetBranch = " + $Branch)
Log ("Repo = " + $RepoOwner + "/" + $RepoName)

Set-Location $ProjectRoot

if (-not (Test-Path ".git")) {
    Fail "This directory is not a git repository"
}

Log ("CurrentDirectory = " + (Get-Location).Path)

$currentBranch = ((Exec-Git @("rev-parse", "--abbrev-ref", "HEAD")) -join "`n").Trim()
if (-not $currentBranch) { Fail "Cannot detect current branch" }
Log ("CurrentBranch = " + $currentBranch)

if ($currentBranch -ne $Branch) {
    Fail ("Wrong branch. Expected: " + $Branch + "; Current: " + $currentBranch)
}

$origin = ((Exec-Git @("remote", "get-url", "origin")) -join "`n").Trim()
if (-not $origin) { Fail "Remote origin not found" }
Log ("Origin = " + $origin)

Check-SensitiveTracked

Log "Git status before pre-pull phase:"
$statusBefore = @(Exec-Git @("status", "--short"))
if ($statusBefore.Count -eq 0) {
    Log "Working tree is clean before sync"
} else {
    foreach ($x in $statusBefore) { Log ("  " + $x) }
}

$porcelain = @(Exec-Git @("status", "--porcelain"))
if ($porcelain.Count -gt 0) {
    Log "Local changes detected before pull. Staging"
    Exec-Git @("add", ".") | Out-Null

    $preStaged = @(Exec-Git @("diff", "--cached", "--name-only"))
    if ($preStaged.Count -gt 0) {
        foreach ($x in $preStaged) { Log ("  pre-pull: " + $x) }
        $preMsg = "auto commit before pull " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        Log ("PrePullCommitMessage = " + $preMsg)
        Exec-Git @("commit", "-m", $preMsg) | Out-Null
        Log "Pre-pull auto-commit created"
    }
} else {
    Log "No local changes before pull"
}

Log "Running git pull --rebase"
$pullOut = @(Exec-Git @("pull", "--rebase", "origin", $Branch))
foreach ($x in $pullOut) { Log ("  " + $x) }
Log "Pull --rebase completed"

Update-RawLinks
Update-RepoMap
Update-AiContext
Update-Status

Log "Staging changes with git add ."
Exec-Git @("add", ".") | Out-Null
Log "git add completed"

$staged = @(Exec-Git @("diff", "--cached", "--name-only"))
if ($staged.Count -eq 0) {
    Log "No staged changes after update. Nothing to commit or push"
    Log ("LogFile = " + $script:LogFile)
    exit 0
}

Log "Staged files:"
foreach ($x in $staged) { Log ("  " + $x) }

if ([string]::IsNullOrWhiteSpace($CommitMessage)) {
    if ($staged.Count -le 8) {
        $CommitMessage = "sync: " + ($staged -join ", ")
    } else {
        $CommitMessage = "sync project updates ($($staged.Count) files)"
    }
}

Log ("CommitMessage = " + $CommitMessage)

Log "Creating commit"
$commitOut = @(Exec-Git @("commit", "-m", $CommitMessage))
foreach ($x in $commitOut) { Log ("  " + $x) }
Log "Commit created"

Log "Pushing to GitHub"
$pushOut = @(Exec-Git @("push", "origin", $Branch))
foreach ($x in $pushOut) { Log ("  " + $x) }
Log "Push completed"

$lastCommit = ((Exec-Git @("log", "-1", "--oneline")) -join "`n").Trim()
Log ("LastCommit = " + $lastCommit)

Log "SYNC SUMMARY"
Log ("SyncedFilesCount = " + $staged.Count)
foreach ($x in $staged) { Log ("Synced: " + $x) }

Log ("LogFile = " + $script:LogFile)
Log "DONE"