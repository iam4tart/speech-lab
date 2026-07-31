import os
import torch

class CheckpointManager:
    def __init__(self, ckpt_dir="checkpoints"):
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.latest_path = os.path.join(self.ckpt_dir, "latest.pt")

    def save(self, model, optimizer, step, epoch, loss):
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "loss": loss,
        }

        tmp_path = self.latest_path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, self.latest_path)  # atomic

        if step % 2000 == 0:
            torch.save(state, os.path.join(self.ckpt_dir, f"step_{step}.pt"))

    def load(self, model, optimizer, device):
        if os.path.exists(self.latest_path):
            ckpt = torch.load(self.latest_path, map_location=device)

            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])

            print(f"Resumed from step {ckpt['step']}")
            return ckpt["step"], ckpt["epoch"]

        print("Starting fresh training")
        return 0, 0
