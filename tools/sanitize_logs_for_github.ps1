# ============================================================
# sanitize_logs_for_github.ps1
# Version: v2.0.0
# Date: 2026-03-07 (Asia/Almaty)
# Safe log sanitizer for GitHub publication
# ============================================================

param(
    [string]$ProjectRoot = "E:\GPT1C_Processor_analitic",
    [string]$SourceDir   = "logs",
    [string]$TargetDir   = "logs_public",
    [switch]$KeepSubfolders,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# ------------------------------------------------------------
# GLOBALS
# ------------------------------------------------------------

$script:ScriptPath = $MyInvocation.MyCommand.Path
$script:ScriptDir  = Split-Path -Parent $script:ScriptPath
$script:RunLog     = Join-Path $script:ScriptDir "sanitize_logs_for_github.runtime.log.txt"

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
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Content
    )

    $dir = Split-Path -Parent $Path
    if ($dir) { Ensure-Dir $dir }

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
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
    Append-Utf8NoBomLine -Path $script:RunLog -Line $line
}

function Fail {
    param([Parameter(Mandatory = $true)][string]$Message)

    Log "ERROR: $Message"
    exit 1
}

function Get-FullPathSafe {
    param([Parameter(Mandatory = $true)][string]$Path)

    return [System.IO.Path]::GetFullPath($Path)
}

function Test-IsSubPath {
    param(
        [Parameter(Mandatory = $true)][string]$ParentPath,
        [Parameter(Mandatory = $true)][string]$ChildPath
    )

    $parent = (Get-FullPathSafe $ParentPath).TrimEnd('\','/') + [System.IO.Path]::DirectorySeparatorChar
    $child  = (Get-FullPathSafe $ChildPath).TrimEnd('\','/') + [System.IO.Path]::DirectorySeparatorChar

    return $child.StartsWith($parent, [System.StringComparison]::OrdinalIgnoreCase)
}

function Assert-PathInsideProject {
    param([Parameter(Mandatory = $true)][string]$Path)

    $root = (Get-FullPathSafe $ProjectRoot).TrimEnd('\','/') + [System.IO.Path]::DirectorySeparatorChar
    $full = (Get-FullPathSafe $Path).TrimEnd('\','/') + [System.IO.Path]::DirectorySeparatorChar

    if (-not $full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        Fail "Path is outside ProjectRoot: $Path"
    }
}

function Get-RepoRelativePath {
    param([Parameter(Mandatory = $true)][string]$AbsolutePath)

    $root = (Get-FullPathSafe $ProjectRoot).TrimEnd('\','/')
    $full = (Get-FullPathSafe $AbsolutePath).TrimEnd('\','/')

    if (-not $full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $null
    }

    $rel = $full.Substring($root.Length).TrimStart('\','/')
    return ($rel -replace "\\","/")
}

function Test-IsProtectedTargetRelative {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $rel = ($RelativePath -replace "\\","/").Trim('/').ToLowerInvariant()

    $protected = @(
        "",
        ".git",
        ".venv",
        "venv",
        "logs",
        "config",
        "reports",
        "cache",
        "archive",
        "tools"
    )

    foreach ($p in $protected) {
        if ($rel -eq $p) { return $true }
        if ($p -ne "" -and $rel.StartsWith($p + "/", [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
    }

    return $false
}

function Assert-SafeDirectories {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    Assert-PathInsideProject -Path $SourcePath
    Assert-PathInsideProject -Path $TargetPath

    $srcFull = Get-FullPathSafe $SourcePath
    $dstFull = Get-FullPathSafe $TargetPath

    $srcRel = Get-RepoRelativePath -AbsolutePath $srcFull
    $dstRel = Get-RepoRelativePath -AbsolutePath $dstFull

    if ([string]::IsNullOrWhiteSpace($srcRel)) {
        Fail "SourceDir resolves to project root. This is forbidden."
    }

    if ([string]::IsNullOrWhiteSpace($dstRel)) {
        Fail "TargetDir resolves to project root. This is forbidden."
    }

    if ($srcFull.TrimEnd('\','/') -ieq $dstFull.TrimEnd('\','/')) {
        Fail "SourceDir and TargetDir must be different"
    }

    if (Test-IsSubPath -ParentPath $srcFull -ChildPath $dstFull) {
        Fail "TargetDir cannot be inside SourceDir"
    }

    if (Test-IsSubPath -ParentPath $dstFull -ChildPath $srcFull) {
        Fail "SourceDir cannot be inside TargetDir"
    }

    if (Test-IsProtectedTargetRelative -RelativePath $dstRel) {
        Fail "TargetDir points to protected project area: $dstRel"
    }

    Log "Directory safety checks passed"
    Log "SourceDir = $srcRel"
    Log "TargetDir = $dstRel"
}

function Test-FileLikelyTextLog {
    param([Parameter(Mandatory = $true)][string]$Path)

    $ext = [System.IO.Path]::GetExtension($Path).ToLowerInvariant()
    $allowed = @(".log", ".txt", ".json", ".ndjson", ".csv")

    if ($allowed -contains $ext) {
        return $true
    }

    return $false
}

function Convert-ToSafeFileName {
    param([Parameter(Mandatory = $true)][string]$Name)

    $safe = $Name -replace '[<>:"/\\|?*]', '_'
    $safe = $safe -replace '\s+', '_'
    return $safe
}

# ------------------------------------------------------------
# SANITIZE RULES
# ------------------------------------------------------------

function Sanitize-Text {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Text)

    $t = $Text

    # Telegram token
    $t = [regex]::Replace($t, '\b\d{8,12}:[A-Za-z0-9_-]{20,}\b', '[REDACTED_TELEGRAM_TOKEN]')

    # Bearer token / authorization
    $t = [regex]::Replace($t, '(?i)\bBearer\s+[A-Za-z0-9\._\-]{12,}\b', 'Bearer [REDACTED]')
    $t = [regex]::Replace($t, '(?i)\bAuthorization\s*[:=]\s*["'']?Bearer\s+[A-Za-z0-9\._\-]{12,}\b', 'Authorization=Bearer [REDACTED]')

    # Common API keys / passwords / secrets
    $t = [regex]::Replace($t, '(?im)\b(OPENAI_API_KEY|DEEPSEEK_API_KEY|TG_BOT_TOKEN|BOT_TOKEN|API_KEY|SECRET_KEY|ACCESS_TOKEN|REFRESH_TOKEN|PASSWORD|PASSWD|PWD|TOKEN)\b\s*[:=]\s*["'']?([^\s"'']+)', '$1=[REDACTED]')
    $t = [regex]::Replace($t, '(?i)\bsk-[A-Za-z0-9]{10,}\b', '[REDACTED_API_KEY]')

    # URLs with secrets
    $t = [regex]::Replace($t, '(?i)(https?://[^\s\?]+)\?([^\s#]*?(token|key|secret|sig|signature|auth)=)([^&#\s]+)', '$1?$2[REDACTED]')

    # Email
    $t = [regex]::Replace($t, '\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b', '[REDACTED_EMAIL]', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)

    # Kazakhstan/other phone-like sequences
    $t = [regex]::Replace($t, '(?<!\d)(?:\+?\d[\d\-\s\(\)]{8,}\d)(?!\d)', '[REDACTED_PHONE]')

    # IPv4
    $t = [regex]::Replace($t, '\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b', '[REDACTED_IP]')

    # chat_id
    $t = [regex]::Replace($t, '(?im)\b(chat_id|admin_chat_id|telegram_chat_id)\b\s*[:=]\s*["'']?(-?\d{5,})', '$1=[REDACTED_CHAT_ID]')

    # Windows user paths
    $t = [regex]::Replace($t, '(?i)\b[A-Z]:\\Users\\[^\\\s]+', 'C:\Users\[REDACTED_USER]')

    # Long hashes / secret-like blobs
    $t = [regex]::Replace($t, '\b[a-f0-9]{32,64}\b', '[REDACTED_HASH]', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)

    return $t
}

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

try {
    Ensure-Dir $script:ScriptDir

    Log "START SANITIZE LOGS"
    Log "ProjectRoot = $ProjectRoot"
    Log "DryRun = $DryRun"
    Log "KeepSubfolders = $KeepSubfolders"

    if (-not (Test-Path -LiteralPath $ProjectRoot)) {
        Fail "ProjectRoot not found: $ProjectRoot"
    }

    $src = Join-Path $ProjectRoot $SourceDir
    $dst = Join-Path $ProjectRoot $TargetDir

    Assert-SafeDirectories -SourcePath $src -TargetPath $dst

    if (-not (Test-Path -LiteralPath $src)) {
        Fail "SourceDir not found: $src"
    }

    Ensure-Dir $dst

    $srcFiles = Get-ChildItem -Path $src -Recurse -File -ErrorAction Stop |
        Where-Object { Test-FileLikelyTextLog -Path $_.FullName } |
        Sort-Object FullName

    Log "Source files detected = $($srcFiles.Count)"

    if ($DryRun) {
        Log "DRY RUN: target cleanup skipped"
    }
    else {
        Log "Cleaning target directory only"
        Get-ChildItem -Path $dst -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction Stop
    }

    $processed = 0
    $skipped   = 0

    foreach ($file in $srcFiles) {
        $srcFile = $file.FullName

        $relativeFromSource = $srcFile.Substring((Get-FullPathSafe $src).TrimEnd('\','/').Length).TrimStart('\','/')
        $relativeFromSource = $relativeFromSource -replace "\\","/"

        if ($KeepSubfolders) {
            $outRel = $relativeFromSource
        }
        else {
            $folderFlat = ($relativeFromSource -replace '/','__')
            $outRel = $folderFlat
        }

        $outRel = Convert-ToSafeFileName -Name $outRel
        $outPath = Join-Path $dst ($outRel -replace '/','\')

        Log "PROCESS: $relativeFromSource"

        $rawText = ""
        try {
            $rawText = Get-Content -LiteralPath $srcFile -Raw -ErrorAction Stop
        }
        catch {
            Log "SKIP unreadable file: $relativeFromSource"
            $skipped++
            continue
        }

        $cleanText = Sanitize-Text -Text $rawText

        if ($DryRun) {
            Log "DRYRUN OUT: $outRel"
        }
        else {
            Write-Utf8NoBomFile -Path $outPath -Content $cleanText
            Log "WROTE: $outRel"
        }

        $processed++
    }

    $manifest = @()
    $manifest += "{"
    $manifest += '  "generated_at": "' + (Get-AlmatyNowString) + '",'
    $manifest += '  "timezone": "Asia/Almaty",'
    $manifest += '  "project_root": "' + (($ProjectRoot -replace '\\','\\') -replace '"','\"') + '",'
    $manifest += '  "source_dir": "' + (($SourceDir -replace '\\','\\') -replace '"','\"') + '",'
    $manifest += '  "target_dir": "' + (($TargetDir -replace '\\','\\') -replace '"','\"') + '",'
    $manifest += '  "dry_run": ' + ($(if ($DryRun) { "true" } else { "false" })) + ','
    $manifest += '  "keep_subfolders": ' + ($(if ($KeepSubfolders) { "true" } else { "false" })) + ','
    $manifest += '  "processed_files": ' + $processed + ','
    $manifest += '  "skipped_files": ' + $skipped
    $manifest += "}"

    if (-not $DryRun) {
        Write-Utf8NoBomFile -Path (Join-Path $dst "_manifest.json") -Content ($manifest -join "`r`n")
        Write-Utf8NoBomFile -Path (Join-Path $dst "_README.txt") -Content @"
This folder contains sanitized copies of logs for public sharing.
Original files in '$SourceDir' were not modified.
Generated at: $(Get-AlmatyNowString) (Asia/Almaty)
"@
    }

    Log "Processed = $processed"
    Log "Skipped = $skipped"
    Log "Original source files were not modified"
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