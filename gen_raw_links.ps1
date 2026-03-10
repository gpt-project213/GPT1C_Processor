# ============================================================
# gen_raw_links.ps1
# GPT1C_Processor / AI 1C PRO
# Генерация raw_links.txt по git ls-files
# Версия: 2.2
# ============================================================

param(
    [string]$Owner = "gpt-project213",
    [string]$Repo = "GPT1C_Processor",
    [string]$Branch = "master",
    [string]$OutputFile = "raw_links.txt"
)

$ErrorActionPreference = "Stop"

Write-Host "Scanning repository..."

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git not found in PATH"
}

$base = "https://raw.githubusercontent.com/$Owner/$Repo/$Branch"

$files = git ls-files
if (-not $files) {
    throw "git ls-files returned empty list"
}

$filtered = $files | Where-Object {

    $_ -ne "raw_links.txt" -and

    (
        $_ -match '\.(py|md|txt|yaml|yml|html)$' -or
        $_ -match '^templates/' -or
        $_ -eq 'config/pattern_config.yaml'
    )

} | Where-Object {

    $_ -notmatch '^\.claude/' -and
    $_ -notmatch '^\.venv/' -and
    $_ -notmatch '^__pycache__/' -and
    $_ -notmatch '^logs/' -and
    $_ -notmatch '^logs_public/' -and
    $_ -notmatch '^reports/' -and
    $_ -notmatch '^archive/' -and
    $_ -notmatch '^cache/' -and
    $_ -notmatch '^state/' -and
    $_ -notmatch '\.log$' -and
    $_ -notmatch '\.lock$' -and
    $_ -notmatch 'runtime\.log' -and
    $_ -notmatch '(^|/).+_state\.json$'
} | Sort-Object -Unique

$links = foreach ($file in $filtered) {
    $normalized = $file -replace '\\','/'
    "$base/$normalized"
}

Write-Host "Writing raw_links.txt..."

$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines(
    (Join-Path (Get-Location) $OutputFile),
    $links,
    $Utf8NoBom
)

Write-Host ""
Write-Host "Done"
Write-Host "Links generated:" $links.Count
Write-Host "Output file:" (Join-Path (Get-Location) $OutputFile)
Write-Host ""