# RUN:
# powershell -ExecutionPolicy Bypass -File "E:\GPT1C_Processor_analitic\tools\sanitize_logs_for_github.ps1"

param(
    [string]$ProjectRoot = "E:\GPT1C_Processor_analitic",
    [string]$SourceDir = "logs",
    [string]$TargetDir = "logs_public"
)

$ErrorActionPreference = "Stop"

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Sanitize-Text {
    param([string]$Text)

    if ($null -eq $Text) { return "" }

    $t = $Text

    # Telegram bot token
    $t = [regex]::Replace($t, '\b\d{8,12}:[A-Za-z0-9_-]{20,}\b', '[TG_BOT_TOKEN]')

    # API keys / bearer
    $t = [regex]::Replace($t, '\bsk-[A-Za-z0-9]{10,}\b', '[API_KEY]')
    $t = [regex]::Replace($t, '\bBearer\s+[A-Za-z0-9._\-]+\b', 'Bearer [TOKEN]')

    # common key=value secrets
    $t = [regex]::Replace($t, '(?i)(password|passwd|pwd|secret|token|api[_-]?key)\s*[:=]\s*([^\s,;]+)', '$1=[REDACTED]')

    # email
    $t = [regex]::Replace($t, '\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b', '[EMAIL]')

    # chat_id or numeric ids
    $t = [regex]::Replace($t, '(?i)\b(chat_id|chatid)\s*[:=]\s*-?\d+\b', '$1=[CHAT_ID]')
    $t = [regex]::Replace($t, '\b\d{8,15}\b', '[ID]')

    # windows paths
    $t = [regex]::Replace($t, '[A-Za-z]:\\[^\r\n\t]*', '[WINDOWS_PATH]')

    return $t
}

function Sanitize-JsonValue {
    param(
        $Value,
        [string]$Path = ""
    )

    if ($null -eq $Value) {
        return $null
    }

    if ($Value -is [string]) {
        $lowerPath = $Path.ToLower()

        # redact by key path
        if ($lowerPath -match 'token|secret|password|passwd|pwd|api[_-]?key') {
            return "[REDACTED]"
        }

        if ($lowerPath -match 'email') {
            return "[EMAIL]"
        }

        if ($lowerPath -match 'chat_?id|telegram_?id|user_?id') {
            return "[ID]"
        }

        return (Sanitize-Text $Value)
    }

    if ($Value -is [int] -or $Value -is [long] -or $Value -is [double] -or $Value -is [decimal] -or $Value -is [float] -or $Value -is [bool]) {
        $lowerPath = $Path.ToLower()
        if ($lowerPath -match 'chat_?id|telegram_?id|user_?id') {
            return "[ID]"
        }
        return $Value
    }

    if ($Value -is [System.Collections.IDictionary]) {
        $copy = [ordered]@{}
        foreach ($k in $Value.Keys) {
            $childPath = if ([string]::IsNullOrWhiteSpace($Path)) { [string]$k } else { "$Path.$k" }
            $copy[$k] = Sanitize-JsonValue -Value $Value[$k] -Path $childPath
        }
        return $copy
    }

    if ($Value -is [System.Collections.IEnumerable] -and -not ($Value -is [string])) {
        $arr = @()
        $i = 0
        foreach ($item in $Value) {
            $childPath = "$Path[$i]"
            $arr += ,(Sanitize-JsonValue -Value $item -Path $childPath)
            $i++
        }
        return $arr
    }

    return $Value
}

$root = $ProjectRoot
$src = Join-Path $root $SourceDir
$dst = Join-Path $root $TargetDir

if (-not (Test-Path $src)) {
    throw "Source logs folder not found: $src"
}

Ensure-Dir $dst

Get-ChildItem -Path $dst -Recurse -Force -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

$files = Get-ChildItem -Path $src -Recurse -File -Force

foreach ($file in $files) {
    $rel = $file.FullName.Substring($src.Length).TrimStart('\')
    $outPath = Join-Path $dst $rel
    $outDir = Split-Path -Parent $outPath
    Ensure-Dir $outDir

    $ext = $file.Extension.ToLower()

    if ($ext -eq ".json") {
        try {
            $raw = Get-Content -Path $file.FullName -Raw -ErrorAction Stop
            $obj = $raw | ConvertFrom-Json -Depth 100 -ErrorAction Stop
            $cleanObj = Sanitize-JsonValue -Value $obj -Path ""
            $cleanJson = $cleanObj | ConvertTo-Json -Depth 100
            Set-Content -Path $outPath -Value $cleanJson -Encoding UTF8
        }
        catch {
            # fallback as text if invalid JSON
            $content = Get-Content -Path $file.FullName -Raw -ErrorAction SilentlyContinue
            $clean = Sanitize-Text $content
            Set-Content -Path $outPath -Value $clean -Encoding UTF8
        }
    }
    else {
        $content = Get-Content -Path $file.FullName -Raw -ErrorAction SilentlyContinue
        $clean = Sanitize-Text $content
        Set-Content -Path $outPath -Value $clean -Encoding UTF8
    }
}

Write-Host "Sanitized logs exported to: $dst"