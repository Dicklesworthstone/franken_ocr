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
    -Version <vX.Y.Z>  Install a specific version (default: latest release)
    -Dir <path>        Install focr.exe into <path> (default: %LOCALAPPDATA%\Programs\focr)
    -OfflineAssetDir   Read the asset and .sha256 from a local directory (requires -Version)
    -Verify            Run "focr robot selftest" after install and report the verdict
    -NoPull            Suppress the post-install model download prompt/guidance
    -Quiet             Suppress non-error output
    -Force             Reinstall even when the same version is present
    -Help              Show usage and exit

  Environment:
    HTTPS_PROXY        HTTPS proxy for downloads (preferred)
    HTTP_PROXY         HTTP proxy for downloads

  Platform: this installer supports native release assets for x86-64 Windows
  (AMD64) and Windows on ARM64, selecting the matching MSVC target.
#>
[CmdletBinding()]
param(
    [string]$Version,
    [string]$Dir,
    [string]$OfflineAssetDir,
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
$script:Asset           = ''
$script:Target          = ''

$script:Quiet               = [bool]$Quiet
$script:Esc                 = [char]27
$script:UseAnsi             = $false
$script:WebArgs             = @{ UseBasicParsing = $true }
$script:OnPath              = $false
$script:PathPersisted       = $false
$script:InstalledVersionStr = ''
$script:InstallLockStream   = $null

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
  -Version <vX.Y.Z>  Install a specific version (default: latest release)
  -Dir <path>        Install focr.exe into <path> (default: %LOCALAPPDATA%\Programs\focr)
  -OfflineAssetDir   Read the exact asset + .sha256 from a local directory (requires -Version)
  -Verify            Run "focr robot selftest" after install and report the verdict
  -NoPull            Suppress the post-install model download prompt/guidance
  -Quiet             Suppress non-error output
  -Force             Reinstall even when the same version is present
  -Help              Show this help and exit

Environment:
  HTTPS_PROXY        HTTPS proxy for downloads (preferred)
  HTTP_PROXY         HTTP proxy for downloads

Platform:
  Supports native release assets for Windows x86-64 (AMD64) and ARM64.
  On macOS or Linux, use the shell installer (install.sh) instead.

After install, download the default model once (about 4.2 GB):  focr pull
Then parse a page with:                                  focr ocr page.png
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

function Resolve-WindowsPlatform {
    param([string]$Arch)
    switch ($Arch) {
        'x86_64' {
            return [pscustomobject]@{
                Arch   = 'x86_64'
                Target = 'x86_64-pc-windows-msvc'
                Asset  = 'focr-x86_64-pc-windows-msvc.exe'
            }
        }
        'arm64' {
            return [pscustomobject]@{
                Arch   = 'arm64'
                Target = 'aarch64-pc-windows-msvc'
                Asset  = 'focr-aarch64-pc-windows-msvc.exe'
            }
        }
        default { return $null }
    }
}

# ============================================================================
# Version resolution and checksum parsing
# ============================================================================
function Resolve-Version {
    if (-not [string]::IsNullOrEmpty($Version)) { return $Version }
    if (-not [string]::IsNullOrEmpty($OfflineAssetDir)) {
        throw '-OfflineAssetDir requires -Version because release discovery is disabled offline.'
    }

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
    } catch {
        throw "Could not resolve the latest release from the GitHub API. Re-run with -Version vX.Y.Z to pin a known release. ($($_.Exception.Message))"
    }

    throw 'The GitHub latest-release API returned no tag. Re-run with -Version vX.Y.Z to pin a known release.'
}

# Release tags are v-prefixed; accept a bare semver from -Version too.
function ConvertTo-NormalizedVersion {
    param([string]$Value)
    if ([string]::IsNullOrEmpty($Value)) { throw 'A release version is required.' }
    if ($Value -cmatch '^[0-9]') { $Value = "v$Value" }
    if ($Value -cnotmatch '^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$') {
        throw "Invalid version '$Value'; expected vX.Y.Z (or a valid semver prerelease)."
    }
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
        $code = $LASTEXITCODE
        if ($code -ne 0) { return $null }
        $line = $out | Select-Object -First 1
        if ($line) { return ([string]$line).Trim() }
        return $null
    } catch {
        return $null
    } finally {
        $ErrorActionPreference = $prev
    }
}

function Get-FocrReportedSemVer {
    param([string]$Value)
    if ([string]::IsNullOrEmpty($Value)) { return $null }
    $m = [regex]::Match($Value, '^focr\s+([0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)$')
    if (-not $m.Success) { return $null }
    return $m.Groups[1].Value
}

function Test-AlreadyInstalled {
    param([string]$Target)
    $exe = Join-Path $script:InstallDir 'focr.exe'
    if (-not (Test-Path $exe)) { return $false }
    $v = Get-FocrVersionString -Exe $exe
    if ([string]::IsNullOrEmpty($v)) { return $false }
    $reported = Get-FocrReportedSemVer -Value $v
    if ([string]::IsNullOrEmpty($reported)) { return $false }
    $want = $Target -creplace '^v', ''
    return ($reported -ceq $want)
}

# ============================================================================
# Cross-process install lock. The handle is opened with FileShare.None in the
# destination directory, so aliases of the same directory and different login
# sessions contend on the same filesystem object. Windows releases the handle
# automatically if the installer exits or crashes; the sentinel file remains.
# ============================================================================
function Enter-InstallLock {
    $lockPath = Join-Path $script:InstallDir '.focr-install.lock'
    try {
        $script:InstallLockStream = [System.IO.File]::Open(
            $lockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    } catch [System.IO.IOException] {
        throw "Another focr installer is already replacing $($script:InstallDir)\focr.exe."
    }
}

function Exit-InstallLock {
    if ($script:InstallLockStream) {
        $script:InstallLockStream.Dispose()
        $script:InstallLockStream = $null
    }
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
    param([string]$Exe, [string]$Version)
    $v = Get-FocrVersionString -Exe $Exe
    if ([string]::IsNullOrEmpty($v)) {
        throw "Installed binary failed its mandatory execution check: $Exe --version"
    }
    $reported = Get-FocrReportedSemVer -Value $v
    $expected = $Version -creplace '^v', ''
    if ([string]::IsNullOrEmpty($reported)) {
        throw "Installed binary returned an invalid version report: '$v'"
    }
    if ($reported -cne $expected) {
        throw "Installed binary version mismatch: expected $expected, reported $reported"
    }
    $script:InstalledVersionStr = $v
    Ok "focr is working: $v"
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
        throw "Self-test could not run: $($_.Exception.Message)"
    } finally {
        $ErrorActionPreference = $prev
    }
    if ($code -eq 0) {
        Ok 'Self-test passed: the int8 kernel matches the scalar oracle on this host.'
    } else {
        throw 'Self-test reported a divergence (see the verdict above).'
    }
}

# Model acquisition. Never auto-download multi-gigabyte weights in quiet or
# non-interactive runs; offer an explicit y/N choice otherwise.
function Invoke-ModelPull {
    param([string]$Exe)
    if ($NoPull) { return }

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
    Info "The download is about 4.2 GB into $($script:ModelCache)."
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
    $lines.Add('  focr pull                 download the default model (about 4.2 GB)')
    $lines.Add('  focr ocr page.png         parse an image into markdown')
    $lines.Add('  focr ocr page.png --json  emit structured JSON (with bounding boxes)')
    $lines.Add('  focr ocr page.png -o out.md    write markdown to a file')
    $lines.Add('  focr ocr page.png -o out.json  write JSON (markdown + boxes) to a file')
    $lines.Add('  focr ocr page.png -o out.md --extract-figures   save figures next to the .md')
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
    $platform = Resolve-WindowsPlatform -Arch $arch
    if ($null -eq $platform) {
        Err "No prebuilt focr binary is available for Windows on '$arch'."
        Err 'Supported Windows architectures: x86-64 (AMD64) and ARM64.'
        Err "Questions: https://github.com/$($script:Owner)/$($script:Repo)/issues"
        return 1
    }
    $script:Asset = $platform.Asset
    $script:Target = $platform.Target

    Initialize-Proxy

    try {
        $version = Resolve-Version
        $version = ConvertTo-NormalizedVersion $version

        $base     = "https://github.com/$($script:Owner)/$($script:Repo)/releases/download/$version"
        if (-not [string]::IsNullOrEmpty($OfflineAssetDir)) {
            $base = [System.IO.Path]::GetFullPath($OfflineAssetDir)
        }
        $assetUrl = "$base/$($script:Asset)"
        $shaUrl   = "$assetUrl.sha256"
        $target   = Join-Path $script:InstallDir 'focr.exe'

        Info "Platform:    windows/$($platform.Arch) ($($script:Target))"
        Info "Asset:       $($script:Asset)"
        Info "Version:     $version"
        Info "Install dir: $($script:InstallDir)"

        if (-not (Test-Path $script:InstallDir)) {
            New-Item -ItemType Directory -Path $script:InstallDir -Force | Out-Null
        }
        Enter-InstallLock

        if ($env:FOCR_INSTALL_TEST_MODE -eq '1') {
            if (-not [string]::IsNullOrEmpty($env:FOCR_INSTALL_TEST_LOCK_READY_PATH)) {
                [System.IO.File]::WriteAllText($env:FOCR_INSTALL_TEST_LOCK_READY_PATH, 'locked')
            }
            if ($env:FOCR_INSTALL_TEST_HOLD_LOCK_SECONDS -match '^[0-9]+$' -and
                [int]$env:FOCR_INSTALL_TEST_HOLD_LOCK_SECONDS -gt 0) {
                Start-Sleep -Seconds ([int]$env:FOCR_INSTALL_TEST_HOLD_LOCK_SECONDS)
            }
        }

        # Keep the version decision under the destination lock, and honor an
        # explicit verification request even when no download is necessary.
        if (-not $Force -and (Test-AlreadyInstalled -Target $version)) {
            Ok "focr $version is already installed at $target"
            Info 'Use -Force to reinstall.'
            Update-Path
            Confirm-Install -Exe $target -Version $version
            if ($Verify) { Invoke-SelfTest -Exe $target }
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
            if (-not [string]::IsNullOrEmpty($OfflineAssetDir)) {
                $offlineAsset = Join-Path $base $script:Asset
                $offlineSha = "$offlineAsset.sha256"
                if (-not (Test-Path -LiteralPath $offlineAsset -PathType Leaf) -or
                    -not (Test-Path -LiteralPath $offlineSha -PathType Leaf)) {
                    throw "Offline asset directory must contain $($script:Asset) and $($script:Asset).sha256."
                }
                Copy-Item -LiteralPath $offlineAsset -Destination $assetPath
                Copy-Item -LiteralPath $offlineSha -Destination $shaPath
            } else {
                try {
                    Invoke-WebRequest -Uri $assetUrl -OutFile $assetPath -TimeoutSec 600 @wa
                } catch {
                    throw "Failed to download $assetUrl ($($_.Exception.Message)). Verify the version exists, or pass -Version to pin a known release."
                }
            }
            if (-not (Test-Path $assetPath) -or ((Get-Item $assetPath).Length -eq 0)) {
                throw "Downloaded file is empty: $($script:Asset)"
            }

            if ([string]::IsNullOrEmpty($OfflineAssetDir)) {
                Info 'Fetching checksum sidecar'
                try {
                    Invoke-WebRequest -Uri $shaUrl -OutFile $shaPath -TimeoutSec 60 @wa
                } catch {
                    throw "Could not fetch the checksum sidecar $shaUrl ($($_.Exception.Message))."
                }
            }

            $expected = Read-ExpectedHash -Path $shaPath
            $actual   = (Get-FileHash -Path $assetPath -Algorithm SHA256).Hash
            if (-not ($actual -ieq $expected)) {
                throw "Checksum mismatch for $($script:Asset). expected $expected, got $actual. The download may be corrupt or tampered with; aborting."
            }
            Ok "Checksum verified ($($actual.Substring(0, 16).ToLowerInvariant())...)"

            $staged = Join-Path $script:InstallDir ('.focr.install.' + [System.Guid]::NewGuid().ToString('N') + '.exe')
            $backup = Join-Path $script:InstallDir ('.focr.backup.' + [System.Guid]::NewGuid().ToString('N') + '.exe')
            try {
                Copy-Item -LiteralPath $assetPath -Destination $staged
                $stagedHash = (Get-FileHash -LiteralPath $staged -Algorithm SHA256).Hash
                if (-not ($stagedHash -ieq $actual)) {
                    throw 'Same-directory staged binary does not match the verified download.'
                }
                $stagedVersion = Get-FocrVersionString -Exe $staged
                $stagedSemVer = Get-FocrReportedSemVer -Value $stagedVersion
                $expectedVersion = $version -creplace '^v', ''
                if ([string]::IsNullOrEmpty($stagedSemVer) -or $stagedSemVer -cne $expectedVersion) {
                    throw "Staged binary failed the version check before replacement: expected $expectedVersion, reported $stagedSemVer."
                }
                if ($env:FOCR_INSTALL_TEST_MODE -eq '1' -and
                    $env:FOCR_INSTALL_TEST_FAILPOINT -eq 'before-replace') {
                    throw 'Injected installer failure before atomic replacement.'
                }
                $hadExistingTarget = Test-Path -LiteralPath $target -PathType Leaf
                if ($hadExistingTarget) {
                    try {
                        if ($env:FOCR_INSTALL_TEST_MODE -eq '1' -and
                            $env:FOCR_INSTALL_TEST_FAILPOINT -eq 'replace-target-missing') {
                            # Model ReplaceFileW error 1177: the old destination
                            # has moved to backup but the replacement did not land.
                            [System.IO.File]::Move($target, $backup)
                            throw 'Injected replacement failure after the target moved to backup.'
                        }
                        [System.IO.File]::Replace($staged, $target, $backup)
                        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
                            throw 'Atomic replacement returned without installing focr.exe.'
                        }
                        $installedHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
                        if (-not ($installedHash -ieq $actual)) {
                            throw 'Installed focr.exe does not match the verified staged binary.'
                        }
                    } catch {
                        $replacementError = $_.Exception
                        if (Test-Path -LiteralPath $backup -PathType Leaf) {
                            $failedReplacement = Join-Path $script:InstallDir ('.focr.failed.' + [System.Guid]::NewGuid().ToString('N') + '.exe')
                            try {
                                if (Test-Path -LiteralPath $target -PathType Leaf) {
                                    [System.IO.File]::Replace($backup, $target, $failedReplacement)
                                } else {
                                    [System.IO.File]::Move($backup, $target)
                                }
                            } catch {
                                $recoveryError = $_.Exception
                                if (-not (Test-Path -LiteralPath $target) -and
                                    (Test-Path -LiteralPath $backup -PathType Leaf)) {
                                    try { [System.IO.File]::Move($backup, $target) } catch { }
                                }
                                if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
                                    throw "Atomic replacement failed and the previous focr.exe could not be restored from $backup ($($recoveryError.Message))."
                                }
                            } finally {
                                if (Test-Path -LiteralPath $failedReplacement -PathType Leaf) {
                                    Remove-Item -LiteralPath $failedReplacement -Force -ErrorAction SilentlyContinue
                                }
                            }
                        }
                        throw $replacementError
                    }
                } else {
                    [System.IO.File]::Move($staged, $target)
                    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
                        throw 'Atomic install returned without installing focr.exe.'
                    }
                    $installedHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
                    if (-not ($installedHash -ieq $actual)) {
                        throw 'Installed focr.exe does not match the verified staged binary.'
                    }
                }
                $staged = $null
                if (Test-Path -LiteralPath $backup -PathType Leaf) {
                    Remove-Item -LiteralPath $backup -Force -ErrorAction SilentlyContinue
                }
            } finally {
                if ($staged -and (Test-Path -LiteralPath $staged)) {
                    Remove-Item -LiteralPath $staged -Force -ErrorAction SilentlyContinue
                }
            }
            Ok "Installed focr to $target"
        } finally {
            $ProgressPreference = $prevProgress
            if ($tmp -and (Test-Path $tmp)) {
                Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
            }
        }

        Update-Path
        Confirm-Install -Exe $target -Version $version

        if ($Verify) { Invoke-SelfTest -Exe $target }

        Invoke-ModelPull -Exe $target
        Write-Summary -Exe $target -Version $version
        return 0
    } catch {
        Err $_.Exception.Message
        return 1
    } finally {
        Exit-InstallLock
    }
}

$exitCode = Main

# When run as a script file, set the process exit code so CI sees failures.
# When fetched via "irm | iex" there is no command path, so do not call exit
# (that would close the user's interactive shell). Set LASTEXITCODE explicitly
# and throw on failure so automation observes a failing pipeline/catchable error.
$runningAsFile = $false
try { $runningAsFile = -not [string]::IsNullOrEmpty($PSCommandPath) } catch { $runningAsFile = $false }

if ($runningAsFile) {
    exit $exitCode
} else {
    $global:LASTEXITCODE = $exitCode
    if ($exitCode -ne 0) {
        throw "focr installer failed with exit code $exitCode."
    }
}
