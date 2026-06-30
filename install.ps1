<#
  focr installer (franken_ocr) for Windows

  focr is a pure-Rust, CPU-only OCR command-line tool. It parses document
  images into structured markdown or JSON using the Baidu Unlimited-OCR
  vision-language model. No Python, no CUDA, no GPU.

  One-liner install:

    irm https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.ps1 | iex

  Cache buster: raw.githubusercontent.com is served through a CDN that can hold
  a stale copy for a few minutes after a push. If the one-liner fetches an old
  script, add a throwaway query string to force a fresh copy:

    irm "https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.ps1?cb=$(Get-Random)" | iex

  Passing options: Invoke-Expression cannot forward arguments, so to use a flag
  download the script into a scriptblock and call it with the flag attached:

    & ([scriptblock]::Create((irm https://raw.githubusercontent.com/Dicklesworthstone/franken_ocr/main/install.ps1))) -Verify

  Options:
    -Version <vX.Y.Z>  Install a specific version (default: latest, falls back to v0.2.0)
    -Dir <path>        Install focr.exe into <path> (default: %LOCALAPPDATA%\Programs\focr)
    -Verify            Run "focr robot selftest" after install and report the verdict
    -NoPull            Do not offer to download the model after install (focr pull)
    -Quiet             Suppress non-error output
    -Force             Reinstall even when the same version is present
    -Help              Show usage and exit

  Environment:
    HTTPS_PROXY        HTTPS proxy for downloads (preferred)
    HTTP_PROXY         HTTP proxy for downloads

  Platform: only x86-64 Windows (AMD64) has a published binary in v0.2.0.
  Windows on ARM64 is not published yet; this installer says so and stops.
#>
[CmdletBinding()]
param(
    [string]$Version,
    [string]$Dir,
    [switch]$Verify,
    [switch]$NoPull,
    [switch]$Quiet,
    [switch]$Force,
    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ============================================================================
# Configuration and runtime state
# ============================================================================
$script:Owner           = 'Dicklesworthstone'
$script:Repo            = 'franken_ocr'
$script:Asset           = 'focr-x86_64-pc-windows-msvc.exe'
$script:FallbackVersion = 'v0.2.0'

$script:Quiet               = [bool]$Quiet
$script:Esc                 = [char]27
$script:UseAnsi             = $false
$script:WebArgs             = @{ UseBasicParsing = $true }
$script:OnPath              = $false
$script:PathPersisted       = $false
$script:InstalledVersionStr = ''

# Negotiate TLS up front. Tls12 is the floor; add Tls13 when the runtime knows
# the value (older .NET on PowerShell 5.1 does not).
try {
    [Net.ServicePointManager]::SecurityProtocol = `
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }
try {
    [Net.ServicePointManager]::SecurityProtocol = `
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls13
} catch { }

# Prefer ANSI color when the host can render it; fall back to Write-Host colors.
try {
    if ($Host -and $Host.UI -and $Host.UI.SupportsVirtualTerminal) {
        $script:UseAnsi = $true
    }
} catch {
    $script:UseAnsi = $false
}

# Resolve the install directory: -Dir wins, otherwise %LOCALAPPDATA%\Programs\focr.
# Guard every Join-Path against a null env var: on non-Windows PowerShell both
# LOCALAPPDATA and USERPROFILE are unset, and `Join-Path $null ...` would throw a
# raw .NET stack trace HERE (top level, under ErrorActionPreference=Stop) — before
# Main's platform check can print the friendly "use install.sh" message. Leaving
# the dir empty lets that guard do its job.
$resolvedDir = $Dir
if ([string]::IsNullOrEmpty($resolvedDir)) {
    $localApp = $env:LOCALAPPDATA
    if ([string]::IsNullOrEmpty($localApp) -and -not [string]::IsNullOrEmpty($env:USERPROFILE)) {
        $localApp = Join-Path $env:USERPROFILE 'AppData\Local'
    }
    if (-not [string]::IsNullOrEmpty($localApp)) {
        $resolvedDir = Join-Path $localApp 'Programs\focr'
    }
}
$script:InstallDir = $resolvedDir

# Model cache resolves to %LOCALAPPDATA%\franken_ocr\models on Windows. Same
# null-env guard so non-Windows pwsh reaches Main's friendly platform check.
$localAppForModel = $env:LOCALAPPDATA
if ([string]::IsNullOrEmpty($localAppForModel) -and -not [string]::IsNullOrEmpty($env:USERPROFILE)) {
    $localAppForModel = Join-Path $env:USERPROFILE 'AppData\Local'
}
if (-not [string]::IsNullOrEmpty($localAppForModel)) {
    $script:ModelCache = Join-Path $localAppForModel 'franken_ocr\models'
} else {
    $script:ModelCache = ''
}

# ============================================================================
# Output helpers: ANSI when supported, Write-Host colors otherwise
# ============================================================================
function Write-Colored {
    param([string]$Text, [string]$AnsiCode, [System.ConsoleColor]$Color)
    if ($script:UseAnsi) {
        Write-Host ("{0}[{1}m{2}{0}[0m" -f $script:Esc, $AnsiCode, $Text)
    } else {
        Write-Host $Text -ForegroundColor $Color
    }
}

function Write-Status {
    param([string]$Tag, [string]$AnsiCode, [System.ConsoleColor]$Color, [string]$Message)
    if ($script:UseAnsi) {
        Write-Host ("{0}[{1}m{2}{0}[0m {3}" -f $script:Esc, $AnsiCode, $Tag, $Message)
    } else {
        Write-Host "$Tag " -ForegroundColor $Color -NoNewline
        Write-Host $Message
    }
}

function Info {
    param([string]$Message)
    if ($script:Quiet) { return }
    Write-Status '->' '0;34' Blue $Message
}

function Ok {
    param([string]$Message)
    if ($script:Quiet) { return }
    Write-Status 'ok' '0;32' Green $Message
}

function Warn {
    param([string]$Message)
    if ($script:Quiet) { return }
    Write-Status 'warn' '0;33' Yellow $Message
}

# Err is never silenced by -Quiet; failures must always be visible.
function Err {
    param([string]$Message)
    Write-Status 'error' '0;31' Red $Message
}

# Draw a framed box around a set of lines, sized to the widest line.
function Write-Box {
    param([string[]]$Lines, [string]$AnsiCode, [System.ConsoleColor]$Color)
    if ($script:Quiet) { return }
    $max = 0
    foreach ($l in $Lines) {
        if ($l.Length -gt $max) { $max = $l.Length }
    }
    $border = '+' + ('=' * ($max + 4)) + '+'
    Write-Colored $border $AnsiCode $Color
    foreach ($l in $Lines) {
        $pad = ' ' * ($max - $l.Length)
        if ($script:UseAnsi) {
            Write-Host ("{0}[{1}m|{0}[0m  {2}{3}  {0}[{1}m|{0}[0m" -f $script:Esc, $AnsiCode, $l, $pad)
        } else {
            Write-Host '|  ' -ForegroundColor $Color -NoNewline
            Write-Host ($l + $pad + '  ') -NoNewline
            Write-Host '|' -ForegroundColor $Color
        }
    }
    Write-Colored $border $AnsiCode $Color
}

function Write-Banner {
    if ($script:Quiet) { return }
    Write-Host ''
    Write-Colored 'focr installer' '1;32' Green
    Write-Colored 'Pure-Rust CPU OCR for document images (franken_ocr)' '0;90' DarkGray
    Write-Host ''
}

# ============================================================================
# Help
# ============================================================================
function Show-Usage {
    $u = @"
focr installer (franken_ocr): pure-Rust CPU OCR for document images

Usage (default install):
  irm https://raw.githubusercontent.com/$($script:Owner)/$($script:Repo)/main/install.ps1 | iex

Usage (with options): load the script into a scriptblock, then pass flags:
  & ([scriptblock]::Create((irm https://raw.githubusercontent.com/$($script:Owner)/$($script:Repo)/main/install.ps1))) -Verify -Dir 'C:\Tools\focr'

Options:
  -Version <vX.Y.Z>  Install a specific version (default: latest, falls back to $($script:FallbackVersion))
  -Dir <path>        Install focr.exe into <path> (default: %LOCALAPPDATA%\Programs\focr)
  -Verify            Run "focr robot selftest" after install and report the verdict
  -NoPull            Do not offer to download the model after install (focr pull)
  -Quiet             Suppress non-error output
  -Force             Reinstall even when the same version is present
  -Help              Show this help and exit

Environment:
  HTTPS_PROXY        HTTPS proxy for downloads (preferred)
  HTTP_PROXY         HTTP proxy for downloads

Platform:
  Only x86-64 Windows (AMD64) has a published binary in $($script:FallbackVersion).
  Windows on ARM64 is not published yet; this installer reports that and stops.
  On macOS or Linux, use the shell installer (install.sh) instead.

After install, download the model once (about 3.9 GB):  focr pull
Then parse a page with:                                 focr ocr page.png
"@
    Write-Host $u
}

# ============================================================================
# Networking helpers
# ============================================================================
function Initialize-Proxy {
    $proxy = $env:HTTPS_PROXY
    if ([string]::IsNullOrEmpty($proxy)) { $proxy = $env:HTTP_PROXY }
    if (-not [string]::IsNullOrEmpty($proxy)) {
        $script:WebArgs['Proxy'] = $proxy
        $script:WebArgs['ProxyUseDefaultCredentials'] = $true
        Info "Using proxy: $proxy"
    }
}

# ============================================================================
# Platform detection
# ============================================================================
function Get-MachineArch {
    # PROCESSOR_ARCHITEW6432 holds the real machine arch when a 32-bit process
    # runs under WOW64; prefer it when present.
    $a = $env:PROCESSOR_ARCHITECTURE
    if (-not [string]::IsNullOrEmpty($env:PROCESSOR_ARCHITEW6432)) {
        $a = $env:PROCESSOR_ARCHITEW6432
    }
    if ([string]::IsNullOrEmpty($a)) { return 'unknown' }
    switch ($a.ToUpperInvariant()) {
        'AMD64' { return 'x86_64' }
        'ARM64' { return 'arm64' }
        'X86'   { return 'x86' }
        default { return $a.ToLowerInvariant() }
    }
}

function Write-Arm64Note {
    if ($script:Quiet) {
        Err "Windows on ARM64 is not a published target for focr $($script:FallbackVersion)."
        return
    }
    $lines = @(
        'Windows on ARM64 is not a published target yet.',
        '',
        "focr $($script:FallbackVersion) ships one Windows binary, built for x86-64 (AMD64).",
        'Windows 11 on ARM can run x64 binaries through emulation, but that',
        'asset is not published, so this installer will not guess for you.',
        '',
        "Track native ARM64 support: https://github.com/$($script:Owner)/$($script:Repo)/issues"
    )
    Write-Host ''
    Write-Colored 'Windows ARM64 is not supported yet' '1;33' Yellow
    Write-Box -Lines $lines -AnsiCode '0;33' -Color Yellow
}

# ============================================================================
# Version resolution and checksum parsing
# ============================================================================
function Resolve-Version {
    if (-not [string]::IsNullOrEmpty($Version)) { return $Version }

    Info 'Resolving the latest release...'
    $api = "https://api.github.com/repos/$($script:Owner)/$($script:Repo)/releases/latest"
    try {
        $headers = @{
            'Accept'     = 'application/vnd.github+json'
            'User-Agent' = 'focr-installer'
        }
        # Copy to a local before splatting (splatting takes an unscoped name).
        $wa = $script:WebArgs
        $rel = Invoke-RestMethod -Uri $api -Headers $headers -TimeoutSec 30 @wa
        $tag = $null
        if ($rel) {
            $prop = $rel.PSObject.Properties['tag_name']
            if ($prop -and $prop.Value) { $tag = [string]$prop.Value }
        }
        if (-not [string]::IsNullOrEmpty($tag)) {
            Info "Latest release: $tag"
            return $tag
        }
    } catch { }

    Warn "Could not resolve the latest release; using $($script:FallbackVersion)"
    return $script:FallbackVersion
}

# Release tags are v-prefixed; accept a bare semver from -Version too.
function ConvertTo-NormalizedVersion {
    param([string]$Value)
    if ([string]::IsNullOrEmpty($Value)) { return $script:FallbackVersion }
    if ($Value -match '^[0-9]') { return "v$Value" }
    return $Value
}

function Read-ExpectedHash {
    param([string]$Path)
    $line = Get-Content -Path $Path -TotalCount 1
    if ([string]::IsNullOrWhiteSpace($line)) {
        throw 'The checksum sidecar was empty.'
    }
    # Sidecar format is "<hex>  <asset>"; take the first whitespace-delimited field.
    $token = ($line.Trim() -split '\s+')[0]
    if ($token -notmatch '^[0-9a-fA-F]{64}$') {
        throw 'The checksum sidecar did not contain a valid SHA256 digest.'
    }
    return $token
}

# ============================================================================
# Installed-version probe (drives the already-installed short-circuit)
# ============================================================================
function Get-FocrVersionString {
    param([string]$Exe)
    if (-not (Test-Path $Exe)) { return $null }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $Exe --version 2>$null
        $line = $out | Select-Object -First 1
        if ($line) { return ([string]$line).Trim() }
        return $null
    } catch {
        return $null
    } finally {
        $ErrorActionPreference = $prev
    }
}

function Test-AlreadyInstalled {
    param([string]$Target)
    $exe = Join-Path $script:InstallDir 'focr.exe'
    if (-not (Test-Path $exe)) { return $false }
    $v = Get-FocrVersionString -Exe $exe
    if ([string]::IsNullOrEmpty($v)) { return $false }
    $m = [regex]::Match($v, '[0-9]+\.[0-9]+\.[0-9]+')
    if (-not $m.Success) { return $false }
    $want = $Target -replace '^v', ''
    return ($m.Value -eq $want)
}

# ============================================================================
# PATH setup (persist to the user PATH only when not already present)
# ============================================================================
function Add-SessionPath {
    param([string]$Dir)
    $cur = $env:Path
    $has = $false
    if (-not [string]::IsNullOrEmpty($cur)) {
        foreach ($p in ($cur -split ';')) {
            if ($p.TrimEnd('\') -ieq $Dir.TrimEnd('\')) { $has = $true; break }
        }
    }
    if (-not $has) {
        if ([string]::IsNullOrEmpty($cur)) { $env:Path = $Dir }
        else { $env:Path = "$cur;$Dir" }
    }
}

function Update-Path {
    $dir = $script:InstallDir
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')

    $needle = $dir.TrimEnd('\')
    $already = $false
    if (-not [string]::IsNullOrEmpty($userPath)) {
        foreach ($p in ($userPath -split ';')) {
            if ([string]::IsNullOrEmpty($p)) { continue }
            if ($p.TrimEnd('\') -ieq $needle) { $already = $true; break }
        }
    }

    if ($already) {
        $script:OnPath = $true
        Add-SessionPath $dir
        Info "$dir is already on your user PATH."
        return
    }

    if ([string]::IsNullOrEmpty($userPath)) { $new = $dir }
    else { $new = ($userPath.TrimEnd(';') + ';' + $dir) }

    [Environment]::SetEnvironmentVariable('Path', $new, 'User')
    Add-SessionPath $dir
    $script:OnPath = $true
    $script:PathPersisted = $true
    Warn "Added $dir to your user PATH. Open a new terminal (or restart your shell) for other sessions to see it."
}

# ============================================================================
# Post-install: version check, optional self-test, model-placement instructions
# ============================================================================
function Confirm-Install {
    param([string]$Exe)
    $v = Get-FocrVersionString -Exe $Exe
    if (-not [string]::IsNullOrEmpty($v)) {
        $script:InstalledVersionStr = $v
        Ok "focr is working: $v"
    } else {
        Warn "Installed the binary, but 'focr --version' returned no output."
        Warn "If $($script:InstallDir) is not on PATH in this shell yet, open a new terminal."
    }
}

function Invoke-SelfTest {
    param([string]$Exe)
    Info 'Running focr robot selftest...'
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $code = 1
    try {
        & $Exe robot selftest | Out-Host
        $code = $LASTEXITCODE
    } catch {
        Warn "Self-test could not run: $($_.Exception.Message)"
        return
    } finally {
        $ErrorActionPreference = $prev
    }
    if ($code -eq 0) {
        Ok 'Self-test passed: the int8 kernel matches the scalar oracle on this host.'
    } else {
        Warn 'Self-test reported a divergence (see the verdict above).'
    }
}

# Model acquisition. `focr pull` downloads the ~3.9 GB int8 weights + tokenizer
# into the model cache. This is verified working on native Windows: the async
# HTTP/TLS send-path that previously surfaced WSAENOTCONN (os error 10057) was
# fixed (bd-15ow). Mirrors the shell installer's maybe_offer_pull: never
# auto-downloads in quiet or non-interactive runs (CI, piped scripts); offers an
# interactive y/N prompt otherwise so the large download is always a choice.
function Invoke-ModelPull {
    param([string]$Exe)
    if ($NoPull) { return }

    # The model is about 3.9 GB. Never auto-download in quiet or non-interactive
    # runs (CI, cron, piped scripts); just leave a clear hint.
    $interactive = $false
    try {
        $interactive = [Environment]::UserInteractive -and -not [Console]::IsInputRedirected
    } catch {
        $interactive = $false
    }
    if ($script:Quiet -or -not $interactive) {
        Info 'Model weights are not bundled. Download them later with: focr pull'
        return
    }

    Write-Host ''
    Info 'focr needs the OCR model before it can parse a page.'
    Info "The download is about 3.9 GB into $($script:ModelCache)."
    $ans = Read-Host 'Download the model now with focr pull? (y/N)'
    if ($ans -match '^(y|yes)$') {
        Info 'Running: focr pull'
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $code = 1
        try {
            & $Exe pull | Out-Host
            $code = $LASTEXITCODE
        } catch {
            Warn "focr pull could not run: $($_.Exception.Message)"
            return
        } finally {
            $ErrorActionPreference = $prev
        }
        if ($code -eq 0) {
            Ok "Model downloaded into $($script:ModelCache)"
        } else {
            Warn 'focr pull did not finish. Retry later with: focr pull'
        }
    } else {
        Info 'Skipped. Download the model later with: focr pull'
    }
}

# ============================================================================
# Final summary
# ============================================================================
function Write-Summary {
    param([string]$Exe, [string]$Version)
    if ($script:Quiet) { return }

    $ver = $Version
    if (-not [string]::IsNullOrEmpty($script:InstalledVersionStr)) {
        $ver = $script:InstalledVersionStr
    }
    $modelParent = Split-Path -Path $script:ModelCache -Parent

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add('focr is installed.')
    $lines.Add('')
    $lines.Add("Version:   $ver")
    $lines.Add("Location:  $Exe")
    $lines.Add('')
    if ($script:PathPersisted) {
        $lines.Add("PATH:      $($script:InstallDir) added to your user PATH.")
        $lines.Add('           Open a new terminal so every shell can find focr.')
        $lines.Add('')
    }
    $lines.Add('First steps:')
    $lines.Add('  focr pull                 download the model (about 3.9 GB)')
    $lines.Add('  focr ocr page.png         parse an image into markdown')
    $lines.Add('  focr ocr page.png --json  emit structured JSON (with bounding boxes)')
    $lines.Add('  focr ocr page.png -o out.md    write markdown to a file')
    $lines.Add('  focr ocr page.png -o out.json  write JSON (markdown + boxes) to a file')
    $lines.Add('  focr robot selftest       verify the int8 kernel on this host')
    $lines.Add('  focr --help               full command reference')
    $lines.Add('')
    $lines.Add("Model cache: $($script:ModelCache)")
    $lines.Add('')
    $lines.Add('Uninstall:')
    $lines.Add("  Remove-Item '$Exe'")
    $lines.Add("  Remove-Item -Recurse '$modelParent'   (removes the downloaded model)")
    $lines.Add("  Then remove '$($script:InstallDir)' from your user PATH.")

    Write-Host ''
    Write-Box -Lines $lines.ToArray() -AnsiCode '0;32' -Color Green
}

# ============================================================================
# Main
# ============================================================================
function Main {
    if ($Help) { Show-Usage; return 0 }

    if ($PSVersionTable.PSVersion.Major -lt 5) {
        Write-Host 'focr installer requires Windows PowerShell 5.1 or newer.'
        return 1
    }

    Write-Banner

    if ($env:OS -ne 'Windows_NT') {
        Err 'This installer targets Windows. On macOS or Linux, use the shell installer (install.sh).'
        return 1
    }

    $arch = Get-MachineArch
    if ($arch -eq 'arm64') {
        Write-Arm64Note
        return 1
    }
    if ($arch -ne 'x86_64') {
        Err "No prebuilt focr binary is available for Windows on '$arch'."
        Err "focr $($script:FallbackVersion) publishes a single Windows binary, built for x86-64 (AMD64)."
        Err "Questions: https://github.com/$($script:Owner)/$($script:Repo)/issues"
        return 1
    }

    Initialize-Proxy

    try {
        $version = Resolve-Version
        $version = ConvertTo-NormalizedVersion $version

        $base     = "https://github.com/$($script:Owner)/$($script:Repo)/releases/download/$version"
        $assetUrl = "$base/$($script:Asset)"
        $shaUrl   = "$assetUrl.sha256"
        $target   = Join-Path $script:InstallDir 'focr.exe'

        Info 'Platform:    windows/x86_64 (x86_64-pc-windows-msvc)'
        Info "Asset:       $($script:Asset)"
        Info "Version:     $version"
        Info "Install dir: $($script:InstallDir)"

        # Already-installed short-circuit (still offers PATH help and a model hint).
        if (-not $Force -and (Test-AlreadyInstalled -Target $version)) {
            Ok "focr $version is already installed at $target"
            Info 'Use -Force to reinstall.'
            Update-Path
            Info 'Model weights are not bundled; download them with: focr pull'
            return 0
        }

        $prevProgress = $ProgressPreference
        $tmp = $null
        try {
            # The Invoke-WebRequest progress bar is very slow in PowerShell 5.1.
            $ProgressPreference = 'SilentlyContinue'

            $tmp = Join-Path $env:TEMP ('focr-install-' + [System.IO.Path]::GetRandomFileName())
            New-Item -ItemType Directory -Path $tmp -Force | Out-Null

            $assetPath = Join-Path $tmp $script:Asset
            $shaPath   = "$assetPath.sha256"
            # Copy to a local before splatting (splatting takes an unscoped name).
            $wa = $script:WebArgs

            Info "Downloading $($script:Asset) ($version)..."
            try {
                Invoke-WebRequest -Uri $assetUrl -OutFile $assetPath -TimeoutSec 600 @wa
            } catch {
                throw "Failed to download $assetUrl ($($_.Exception.Message)). Verify the version exists, or pass -Version to pin a known release."
            }
            if (-not (Test-Path $assetPath) -or ((Get-Item $assetPath).Length -eq 0)) {
                throw "Downloaded file is empty: $($script:Asset)"
            }

            Info 'Fetching checksum sidecar'
            try {
                Invoke-WebRequest -Uri $shaUrl -OutFile $shaPath -TimeoutSec 60 @wa
            } catch {
                throw "Could not fetch the checksum sidecar $shaUrl ($($_.Exception.Message))."
            }

            $expected = Read-ExpectedHash -Path $shaPath
            $actual   = (Get-FileHash -Path $assetPath -Algorithm SHA256).Hash
            if (-not ($actual -ieq $expected)) {
                throw "Checksum mismatch for $($script:Asset). expected $expected, got $actual. The download may be corrupt or tampered with; aborting."
            }
            Ok "Checksum verified ($($actual.Substring(0, 16).ToLowerInvariant())...)"

            if (-not (Test-Path $script:InstallDir)) {
                New-Item -ItemType Directory -Path $script:InstallDir -Force | Out-Null
            }
            Copy-Item -Path $assetPath -Destination $target -Force
            Ok "Installed focr to $target"
        } finally {
            $ProgressPreference = $prevProgress
            if ($tmp -and (Test-Path $tmp)) {
                Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
            }
        }

        Update-Path
        Confirm-Install -Exe $target

        if ($Verify) { Invoke-SelfTest -Exe $target }

        Invoke-ModelPull -Exe $target
        Write-Summary -Exe $target -Version $version
        return 0
    } catch {
        Err $_.Exception.Message
        return 1
    }
}

$exitCode = Main

# When run as a script file, set the process exit code so CI sees failures.
# When fetched via "irm | iex" there is no command path, so do not call exit
# (that would close the user's interactive shell); record LASTEXITCODE instead.
$runningAsFile = $false
try { $runningAsFile = -not [string]::IsNullOrEmpty($PSCommandPath) } catch { $runningAsFile = $false }

if ($runningAsFile) {
    exit $exitCode
} elseif ($exitCode -ne 0) {
    $global:LASTEXITCODE = $exitCode
}
