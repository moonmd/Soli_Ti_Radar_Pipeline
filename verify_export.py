import numpy as np
import os

EXPORT_PREFIX = os.path.join("dumps", "adc_exported_frame")
IMPORT_PREFIX = os.path.join("dumps", "adc_imported_frame")
FRAME_INDICES = [0, 9, 19, 29, 39]


def compare_adc_frames():
    all_match = True
    for idx in FRAME_INDICES:
        export_file = f"{EXPORT_PREFIX}{idx}.npy"
        import_file = f"{IMPORT_PREFIX}{idx}.npy"
        if not os.path.exists(export_file):
            print(f"[ERROR] Exported file missing: {export_file}")
            all_match = False
            continue
        if not os.path.exists(import_file):
            print(f"[ERROR] Imported file missing: {import_file}")
            all_match = False
            continue
        adc_export = np.load(export_file)
        adc_import = np.load(import_file)
        if adc_export.shape != adc_import.shape:
            print(f"[FAIL] Shape mismatch for frame {idx}: exported {adc_export.shape}, imported {adc_import.shape}")
            all_match = False
            continue
        if np.array_equal(adc_export, adc_import):
            print(f"[OK] Frame {idx}: Exported and imported ADC data match exactly.")
        else:
            diff = np.abs(adc_export.astype(np.int32) - adc_import.astype(np.int32))
            max_diff = diff.max()
            num_diff = np.count_nonzero(diff)
            print(f"[FAIL] Frame {idx}: Data mismatch. Max abs diff: {max_diff}, differing elements: {num_diff}/{adc_export.size}")
            all_match = False
    if all_match:
        print("\nAll compared ADC frames match!\n")
        return 0
    else:
        print("\nSome ADC frames do not match. See details above.\n")
        return 1


if __name__ == "__main__":
    exit(compare_adc_frames())
