import torch
from lightning.fabric import Fabric


def test_inplace():
    fabric = Fabric(accelerator="cpu", devices=1)
    fabric.launch()

    t = torch.tensor([1.0])
    original_ptr = t.data_ptr()

    print(f"Original value: {t.item()}")

    # Execute all_reduce
    result = fabric.all_reduce(t, reduce_op="sum")

    print(f"Returned value: {result.item()}")
    print(f"Was original Tensor modified? (value): {t.item()}")

    if t.data_ptr() != result.data_ptr():
        print(
            "Conclusion: Fabric returned a new Tensor! The original Tensor was not modified in-place (or should not be relied upon)."
        )
        print("Must write as: t = fabric.all_reduce(t)")
    else:
        print(
            "Conclusion: It was modified in-place? (This might be special on single-device CPU, but usually not in distributed settings)"
        )


if __name__ == "__main__":
    test_inplace()
