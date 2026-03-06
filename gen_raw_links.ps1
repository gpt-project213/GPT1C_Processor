# gen_raw_links.ps1
# v1.1.0 · 2026-03-06 (Asia/Almaty)

param(
    [string]$RepoOwner = "gpt-project213",
    [string]$RepoName  = "GPT1C_Processor",
    [string]$Branch    = "master",
    [string]$OutFile   = "raw_links.txt"
)

$ErrorActionPreference = "Stop"

try {

    $base = "https://raw.githubusercontent.com/$RepoOwner/$RepoName/$Branch"

    Write-Host ""
    Write-Host "Repository:"
    Write-Host "$RepoOwner/$RepoName"
    Write-Host ""

    $files = git ls-files

    if (-not $files) {
        throw "git ls-files returned empty result. Run script inside git repository."
    }

    $lines = @()

    $lines += "# RAW links generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    $lines += "# Repository: https://github.com/$RepoOwner/$RepoName"
    $lines += "# Branch: $Branch"
    $lines += ""

    foreach ($file in $files) {

        $normalized = $file -replace "\\", "/"
        $raw = "$base/$normalized"

        $lines += $raw
    }

    Set-Content -Path $OutFile -Value $lines -Encoding UTF8

    Write-Host ""
    Write-Host "RAW links file created:"
    Write-Host $OutFile
    Write-Host ""
    Write-Host "Total RAW links:" $files.Count
    Write-Host ""

}
catch {

    Write-Error $_
    exit 1

}