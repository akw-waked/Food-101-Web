import gdown
import os

def download_model(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if not os.path.exists(output):
        print(f"Downloading {output}...")
        gdown.download(url, output, quiet=False)
    else:
        print(f"{output} already exists.")

def download_all_models():
    models = {
        "models/baseline/food101_checkpoint_best.pth": "1QkRUleJcn_5sKiuN2SXyJXJ3XGrg2QCE",
        "models/pretrained_freeze/food101_checkpoint_best.pth": "1pO76sPOStJ2fXq7C0kB_eHp9dUvFiURZ",
        "models/pretrained_unfreeze/food101_checkpoint_best.pth": "13OQ38gMW_cIRsZ7sYalLp6iOL3bfGiKm",
    }

    for output, file_id in models.items():
        download_model(file_id, output)

# Call the function
download_all_models()
