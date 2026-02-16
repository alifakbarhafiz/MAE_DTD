"""
Run the same steps as colab_mae_dtd.ipynb locally (smoke test).
Execute from project root: python run_colab_local.py
"""
import os
import subprocess
import sys

if __name__ == "__main__":
    # 1. Setup project root
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, PROJECT_DIR)
    print("CWD:", os.getcwd())

    # 2. Install dependencies
    req_path = os.path.join(PROJECT_DIR, "requirements.txt")
    if os.path.isfile(req_path):
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path], cwd=PROJECT_DIR)
        print("Dependencies ready.")
    else:
        print("requirements.txt not found, skipping pip install.")

    # 3. Download DTD
    print("Downloading DTD...")
    from torchvision.datasets import DTD
    os.makedirs("data", exist_ok=True)
    for split in ["train", "val", "test"]:
        DTD(root="data", split=split, partition=1, download=True)
    print("DTD ready.")

    # 4. Config and train (1 epoch smoke test)
    import yaml
    import torch
    from src.utils.config import load_config
    from src.training.train_mae import train_mae

    CONFIG = "configs/mask75.yaml"
    EPOCHS = 1
    cfg = load_config(CONFIG)
    cfg["training"] = cfg.get("training", {}) | {"epochs": EPOCHS}
    with open("configs/colab_run.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    ckpt_dir = train_mae("configs/colab_run.yaml", device=device)
    print("Checkpoints:", ckpt_dir)
    print("Local run (smoke test) finished.")
