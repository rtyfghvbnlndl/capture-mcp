<#
  capture-mcp installer for Windows (PowerShell 5.1+).

  .\install.ps1                 # install to %USERPROFILE%\.capture-mcp\venv
  .\install.ps1 -Ocr            # also install read_screen (screen as text, uses the OCR built into Windows)
  .\install.ps1 -TextOnly       # for models without vision: OCR + CAPTURE_TEXT_ONLY=1 in the client config
  .\install.ps1 -Venv C:\venvs\capmcp

  Prints the MCP client config at the end and writes it to mcp-config.json next to this script.
#>
param(
    [string]$Venv = (Join-Path $env:USERPROFILE ".capture-mcp\venv"),
    [string]$Python = "python",
    [switch]$Ocr,
    [switch]$TextOnly
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# 1. Python >= 3.10
$ver = & $Python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if (-not $ver) { throw "Python not found. Install Python 3.10+ or pass -Python <path to python.exe>." }
$major, $minor = $ver.Split(".") | ForEach-Object { [int]$_ }
if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) { throw "Python $ver is too old; need 3.10+." }
Write-Host "Python $ver"

# 2. Long venv paths break OpenCV's DLL loading on Windows ("filename or extension is too long").
if ($Venv.Length -gt 80) { Write-Warning "Venv path is long ($($Venv.Length) chars); OpenCV may fail to load. Prefer a short path." }

# 3. venv + package
if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    Write-Host "Creating venv at $Venv"
    & $Python -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}
$py = Join-Path $Venv "Scripts\python.exe"
if ($TextOnly) { $Ocr = $true }
$extras = if ($Ocr) { "enum,ocr" } else { "enum" }
Write-Host "Installing capture-mcp[$extras] ..."
& $py -m pip install --disable-pip-version-check -q "$here[$extras]"
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# 4. smoke test: imports + device list (does not open any camera)
& $py -c "import cv2, mcp, capture_mcp.server; print('import OK, OpenCV', cv2.__version__)"
if ($LASTEXITCODE -ne 0) { throw "import check failed" }
Write-Host ""
Write-Host "Video devices:"
& $py -m capture_mcp --list
Write-Host ""
if ($Ocr) {
    Write-Host "OCR self-test:"
    & $py -m capture_mcp --ocr-test
    Write-Host ""
}

# 5. client config
$config = [ordered]@{ mcpServers = [ordered]@{ "capture-card" = [ordered]@{ command = $py; args = @("-m", "capture_mcp"); env = $(if ($TextOnly) { [ordered]@{ CAPTURE_TEXT_ONLY = "1" } } else { @{} }) } } }
$json = $config | ConvertTo-Json -Depth 5
$out = Join-Path $here "mcp-config.json"
[System.IO.File]::WriteAllText($out, $json, (New-Object System.Text.UTF8Encoding $false))
Write-Host "Done. Add this to your MCP client config (also saved to $out):"
Write-Host $json
