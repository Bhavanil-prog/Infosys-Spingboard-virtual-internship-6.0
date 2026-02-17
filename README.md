# Environment setup for AirFly Insights project

Steps to create a Python virtual environment and install dependencies on Windows:

1. Open PowerShell in the project folder: `c:\projectss\Infosys SpringBoard internship 6.0`
2. (Optional) If using PowerShell without script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

3. Run the setup script (creates a `.venv` and installs packages):

```powershell
.\setup_env.ps1
```

Notes:
- The file `requirements.txt` lists the Python packages used for data processing, visualization, and geospatial analysis.
- Some geospatial libraries (for example `geopandas`, `fiona`, `rtree`) can be difficult to install with plain pip on Windows. If you hit issues, I recommend using Anaconda/Miniconda and running:

```powershell
conda create -n airfly python=3.10 -y
conda activate airfly
conda install -c conda-forge geopandas rtree fiona pyproj shapely -y
pip install -r requirements.txt
```

Next steps I can take for you:
- Run the setup script here to install packages (I can attempt it now).
- Create a Jupyter notebook template for data cleaning and exploration.
- Scaffold preprocessing code (sampling, dtype optimizations, feature engineering).

Files added:
- [requirements.txt](requirements.txt)
- [setup_env.ps1](setup_env.ps1)
- [README.md](README.md)

If you want me to run the setup script now, say "Run setup" and I'll execute it in the terminal.
