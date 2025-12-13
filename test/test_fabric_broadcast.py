import torch
import os
from lightning.fabric import Fabric


def main():
    # Initialize Fabric
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        fabric = Fabric(
            devices=2,
            accelerator=accelerator,
            strategy="ddp" if accelerator == "cuda" else "auto",
        )
        fabric.launch()
    except Exception as e:
        print(f"Failed to launch Fabric: {e}")
        return

    rank = fabric.global_rank

    # Rank 0 has 100.0, Rank 1 has 200.0
    val = 100.0 if rank == 0 else 200.0
    t_bc = torch.tensor([val]).to(fabric.device)

    fabric.barrier()

    if rank == 1:
        print(f"[Rank 1] Before Broadcast: {t_bc.item()}")

    # Broadcast from 0 to 1
    result_bc = fabric.broadcast(t_bc, src=0)

    if rank == 1:
        print(f"[Rank 1] After Broadcast (Original Tensor): {t_bc.item()}")
        print(f"[Rank 1] Result Broadcast (Returned Tensor): {result_bc.item()}")

        if abs(t_bc.item() - 100.0) < 1e-5:
            print("CONCLUSION: broadcast IS IN-PLACE")
        else:
            print("CONCLUSION: broadcast IS NOT IN-PLACE")


if __name__ == "__main__":
    main()
