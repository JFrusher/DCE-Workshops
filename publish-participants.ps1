# publish-participants.ps1
# Syncs approved participant files to the public DCE-Workshops repo and pushes.
# Run this any time you update workshop materials.

$private = "c:\Projects\DCE Code"
$public  = "c:\Projects\DCE-Code-Workshops"

if (-not (Test-Path $public)) {
    Write-Error "Public repo not found at $public. Clone dce-code-workshops there first."
    exit 1
}

# Clear public repo content (preserve .git/ and README.md)
Get-ChildItem $public -Exclude ".git", "README.md" | Remove-Item -Recurse -Force

# Helper: copies one file, creating destination directories as needed
function Copy-PublicFile($relPath) {
    $src  = Join-Path $private $relPath
    $dest = Join-Path $public  $relPath
    $dir  = Split-Path $dest -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
    if (-not (Test-Path $src)) { Write-Warning "Source not found, skipping: $relPath"; return }
    Copy-Item $src $dest
}

# --- analytics_workshop ---
Copy-PublicFile "analytics_workshop\participant_analytics.ipynb"
Copy-PublicFile "analytics_workshop\snippets.ipynb"
Copy-PublicFile "analytics_workshop\turbine_telemetry.csv"
Copy-PublicFile "analytics_workshop\wind_turbine_telemetry.csv"
Copy-PublicFile "analytics_workshop\telemetry_schema.md"
Copy-PublicFile "analytics_workshop\Data Analytics & Visualisation in Data-Centric Engineering.pdf"
Copy-PublicFile "analytics_workshop\solutions\solution_analytics.ipynb"

# --- ML+AI_workshop (participants follow Presentation.ipynb live) ---
Copy-PublicFile "ML+AI_workshop\Presentation.ipynb"
Copy-PublicFile "ML+AI_workshop\presentation_pump_telemetry.csv"

# --- prosthetics_workshop ---
Copy-PublicFile "prosthetics_workshop\participant_prosthetics.ipynb"
Copy-PublicFile "prosthetics_workshop\snippets.ipynb"
Copy-PublicFile "prosthetics_workshop\prosthetics_data.csv"
Copy-PublicFile "prosthetics_workshop\solutions\solution_prosthetics.ipynb"

# --- Root ---
Copy-PublicFile "requirements.txt"

# Commit and push
Set-Location $public
git add -A
$date = Get-Date -Format "yyyy-MM-dd"
git commit -m "Update participant materials $date"
git push
Write-Host "Published successfully." -ForegroundColor Green
