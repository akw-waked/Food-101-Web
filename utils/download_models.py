import os
import gdown

def create_model_dirs():
    os.makedirs("models/baseline", exist_ok=True)
    os.makedirs("models/pretrained_freeze", exist_ok=True)
    os.makedirs("models/pretrained_unfreeze", exist_ok=True)

def download_model(url, output):
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

def download_all_models():
    create_model_dirs()
    download_model("https://drive.google.com/uc?id=1QkRUleJcn_5sKiuN2SXyJXJ3XGrg2QCE", "models/baseline/food101_checkpoint_best.pth")
    download_model("https://drive.google.com/uc?id=1pO76sPOStJ2fXq7C0kB_eHp9dUvFiURZ", "models/pretrained_freeze/food101_checkpoint_best.pth")
    download_model("https://drive.google.com/uc?id=13OQ38gMW_cIRsZ7sYalLp6iOL3bfGiKm", "models/pretrained_unfreeze/food101_checkpoint_best.pth")
