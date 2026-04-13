$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

if (-Not (Test-Path -Path ".venv")) {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3.11 -m venv .venv
  } else {
    Write-Host "Warning: 'py' launcher not found. Using default python on PATH."
    python -m venv .venv
  }
}

. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r .\requirements.txt
