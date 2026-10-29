$ErrorActionPreference = "Continue"
$projectDir = "C:\Users\MUNNA\.gemini\antigravity\scratch\smart-expense-tracker"
Set-Location $projectDir

# 1. Kill any existing instances
Get-Process -Name "python", "cloudflared" -ErrorAction SilentlyContinue | Where-Object { 
    $_.Path -like "*tracker*" -or $_.Path -like "*cloudflared*" 
} | Stop-Process -Force -ErrorAction SilentlyContinue

# 2. Start Flask with distinct stdout and stderr
$flaskOut = Join-Path $projectDir "flask_out.log"
$flaskErr = Join-Path $projectDir "flask_err.log"
if (Test-Path $flaskOut) { Remove-Item $flaskOut -Force }
if (Test-Path $flaskErr) { Remove-Item $flaskErr -Force }

$flaskProc = Start-Process `
    -FilePath "C:\ProgramData\tracker_venv\Scripts\python.exe" `
    -ArgumentList "-u app.py" `
    -WorkingDirectory $projectDir `
    -RedirectStandardOutput $flaskOut `
    -RedirectStandardError $flaskErr `
    -PassThru `
    -WindowStyle Hidden

Start-Sleep -Seconds 4

# 3. Start Cloudflare Tunnel
$tunnelLog = Join-Path $projectDir "tunnel.log"
if (Test-Path $tunnelLog) { Remove-Item $tunnelLog -Force }

$tunnelProc = Start-Process `
    -FilePath "C:\ProgramData\cloudflared.exe" `
    -ArgumentList "tunnel --url http://127.0.0.1:5000" `
    -WorkingDirectory $projectDir `
    -RedirectStandardError $tunnelLog `
    -PassThru `
    -WindowStyle Hidden

# 4. Wait for URL
$tunnelUrl = ""
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Path $tunnelLog) {
        $content = Get-Content $tunnelLog -Raw -ErrorAction SilentlyContinue
        if ($content -match "https://[a-zA-Z0-9-]+\.trycloudflare\.com") {
            $tunnelUrl = $matches[0]
            break
        }
    }
}

Write-Output "FLASK_PID: $($flaskProc.Id)"
Write-Output "TUNNEL_PID: $($tunnelProc.Id)"
Write-Output "LIVE_PUBLIC_URL: $tunnelUrl"
