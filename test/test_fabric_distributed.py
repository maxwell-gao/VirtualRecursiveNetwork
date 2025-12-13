import torch
import os
from lightning.fabric import Fabric


def main():
    # Initialize Fabric
    # When launched with torchrun, it detects the environment.
    # We force devices=2 to match torchrun --nproc_per_node=2
    fabric = Fabric(devices=2)
    fabric.launch()

    rank = fabric.global_rank
    world_size = fabric.world_size

    if rank == 0:
        print(f"World Size: {world_size}")
        print("----------------------------------------------------------------")

    # Setup tensor: Rank 0 -> 1.0, Rank 1 -> 2.0, etc.
    val = float(rank + 1)
    t = (
        torch.tensor([val]).cuda()
        if fabric.device.type == "cuda"
        else torch.tensor([val])
    )

    original_ptr = t.data_ptr()
    original_val = t.item()

    print(f"[Rank {rank}] Before: Value={original_val}, Ptr={original_ptr}")

    # Perform all_reduce (Sum)
    # Expected sum for N procs: N*(N+1)/2
    # e.g. 2 procs: 1+2=3
    # e.g. 8 procs: 36
    result = fabric.all_reduce(t, reduce_op="sum")

    # Check if t was modified
    new_val = t.item()
    result_val = result.item()

    print(f"[Rank {rank}] After:  Value={new_val}, Ptr={t.data_ptr()}")
    print(f"[Rank {rank}] Result: Value={result_val}, Ptr={result.data_ptr()}")

    expected_sum = sum(range(1, world_size + 1))

    if abs(new_val - expected_sum) < 1e-5:
        print(f"[Rank {rank}] ==> Input Tensor WAS modified in-place.")
    else:
        print(
            f"[Rank {rank}] ==> Input Tensor was NOT modified. You MUST use the return value."
        )

    if rank == 0:
        print("----------------------------------------------------------------")
        if abs(new_val - expected_sum) > 1e-5:
            print("CONCLUSION: Fabric.all_reduce is NOT in-place. The fix is REQUIRED.")
        else:
            print(
                "CONCLUSION: Fabric.all_reduce IS in-place (unexpected but possible)."
            )


if __name__ == "__main__":
    main()
