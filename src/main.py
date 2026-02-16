"""
Entry point: pretrain, reconstruct, spectrum, linear_probe.
Usage:
  python -m src.main pretrain --config configs/mask75.yaml
  python -m src.main reconstruct --config configs/mask75.yaml --checkpoint experiments/checkpoints/mask75/best.pt
  python -m src.main spectrum --config configs/mask75.yaml --checkpoint experiments/checkpoints/mask75/best.pt
  python -m src.main linear_probe --config configs/mask75.yaml --checkpoint experiments/checkpoints/mask75/best.pt
"""
import argparse
import torch
from pathlib import Path

from .utils.config import load_config
from .training.train_mae import train_mae
from .training.linear_probe import run_linear_probe
from .evaluation.reconstruction import run_reconstruction
from .evaluation.spectrum import run_spectrum_analysis


def main():
    parser = argparse.ArgumentParser(description="MAE on DTD")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pretrain = sub.add_parser("pretrain", help="Pretrain MAE")
    p_pretrain.add_argument("--config", type=str, required=True, help="Path to config YAML")
    p_pretrain.add_argument("--device", type=str, default=None)

    p_recon = sub.add_parser("reconstruct", help="Run reconstruction evaluation")
    p_recon.add_argument("--config", type=str, required=True)
    p_recon.add_argument("--checkpoint", type=str, required=True)
    p_recon.add_argument("--output-dir", type=str, default=None)
    p_recon.add_argument("--device", type=str, default=None)

    p_spec = sub.add_parser("spectrum", help="Run embedding spectrum analysis")
    p_spec.add_argument("--config", type=str, required=True)
    p_spec.add_argument("--checkpoint", type=str, required=True)
    p_spec.add_argument("--output-dir", type=str, default=None)
    p_spec.add_argument("--device", type=str, default=None)

    p_probe = sub.add_parser("linear_probe", help="Run linear probe")
    p_probe.add_argument("--config", type=str, required=True)
    p_probe.add_argument("--checkpoint", type=str, required=True)
    p_probe.add_argument("--epochs", type=int, default=50)
    p_probe.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.command == "pretrain":
        ckpt_dir = train_mae(args.config, device=device)
        print(f"Checkpoints saved to {ckpt_dir}")

    elif args.command == "reconstruct":
        cfg = load_config(args.config)
        paths = cfg.get("paths", {})
        run_name = cfg.get("run_name", "mae")
        output_dir = args.output_dir or Path(paths.get("results_dir", "experiments/results")) / "reconstruction" / run_name
        metrics = run_reconstruction(
            args.config,
            args.checkpoint,
            output_dir,
            device=device,
        )
        print("Reconstruction metrics:", metrics)

    elif args.command == "spectrum":
        cfg = load_config(args.config)
        paths = cfg.get("paths", {})
        run_name = cfg.get("run_name", "mae")
        output_dir = args.output_dir or Path(paths.get("results_dir", "experiments/results")) / "spectrum" / run_name
        metrics = run_spectrum_analysis(
            args.config,
            args.checkpoint,
            output_dir,
            device=device,
        )
        print("Spectrum metrics:", metrics)

    elif args.command == "linear_probe":
        acc = run_linear_probe(
            args.config,
            args.checkpoint,
            device=device,
            epochs=args.epochs,
        )
        print(f"Linear probe test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
