# PowerShell script to create a virtual environment and install packages on Windows
param(
    [string]$VenvDir = ".venv"
)

Write-Host "Creating virtual environment in $VenvDir..."
python -m venv $VenvDir

Write-Host "Activating virtual environment..."
if (Test-Path "$VenvDir\Scripts\Activate.ps1") {
    & "$VenvDir\Scripts\Activate.ps1"
} else {
    Write-Host "Activate script not found. Ensure Python is on PATH and venv creation succeeded." -ForegroundColor Yellow
}

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing packages from requirements.txt (this may take several minutes)..."
python -m pip install -r requirements.txt

Write-Host "Note: geospatial packages like geopandas/fiona/rtree may require system dependencies and conda for smooth install."
Write-Host "If installation fails, consider using Anaconda/Miniconda and run: `conda install -c conda-forge geopandas rtree fiona`" -ForegroundColor Yellow

Write-Host "Installation finished. To activate the venv in this session run:`\n    .\\$VenvDir\\Scripts\\Activate.ps1`" -ForegroundColor Green