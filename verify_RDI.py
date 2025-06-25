import numpy as np
import os

RDI_TYPES = [
    (os.path.join("dumps", "rdi_gen_frame"), "Generator Injected RDI (adc derived)"),
    (os.path.join("dumps", "rdi_raw_frame"), "Raw FFT-based RDI (pre-pipeline)")
]
FRAME_INDICES = [0, 9, 19, 29, 39]


def compare_rdi_dumps():
    all_match = True
    for prefix, desc in RDI_TYPES:
        print(f"\n--- Comparing {desc} ---")
        for idx in FRAME_INDICES:
            gen_file = f"{prefix}{idx}.npy"
            con_file = f"{prefix}{idx}.npy"
            if not os.path.exists(gen_file):
                print(f"[ERROR] Missing file: {gen_file}")
                all_match = False
                continue
            if not os.path.exists(con_file):
                print(f"[ERROR] Missing file: {con_file}")
                all_match = False
                continue
            gen_rdi = np.load(gen_file)
            con_rdi = np.load(con_file)
            # Print stats for generator and consumer RDI
            print(f"  [GEN] Frame {idx}: min={gen_rdi.min():.4f}, max={gen_rdi.max():.4f}, avg={gen_rdi.mean():.4f}, pow={np.mean(gen_rdi**2):.4f}")
            print(f"  [CON] Frame {idx}: min={con_rdi.min():.4f}, max={con_rdi.max():.4f}, avg={con_rdi.mean():.4f}, pow={np.mean(con_rdi**2):.4f}")
            if gen_rdi.shape != con_rdi.shape:
                print(f"[FAIL] Frame {idx}: Shape mismatch: generator {gen_rdi.shape}, consumer {con_rdi.shape}")
                all_match = False
                continue
            diff = np.abs(gen_rdi - con_rdi)
            max_diff = diff.max()
            mean_diff = diff.mean()

            if np.allclose(gen_rdi, con_rdi, atol=1e-3):
                print(f"[OK] Frame {idx}: RDI match within tolerance.")
            else:
                num_diff = np.count_nonzero(diff > 1e-3)
                print(f"[FAIL] Frame {idx}: RDI mismatch. Max abs diff: {max_diff:.4f}, mean abs diff: {mean_diff:.4f}, differing elements: {num_diff}/{gen_rdi.size}")
                all_match = False
    if all_match:
        print("\nAll compared RDI frames match within tolerance!\n")
        return 0
    else:
        print("\nSome RDI frames do not match. See details above.\n")
        return 1


if __name__ == "__main__":
    exit(compare_rdi_dumps())
