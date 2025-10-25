import re
from pathlib import Path
import numpy as np

root = Path(".")
prefix = "encoded_tinystories_"          # adjust if needed
out = root / "encoded_tinystories_merged.npy"

# collect and sort by the `{i}` number
files = []
for p in root.glob(f"{prefix}[0-9]*.npy"):
    m = re.search(rf"{re.escape(prefix)}(\d+)\.npy$", p.name)
    if m and p.name != out.name:
        files.append((int(m.group(1)), p))
files.sort(key=lambda x: x[0])
paths = [p for _, p in files]
print(paths)

arrays = [np.load(p) for p in paths]
# sanity checks (same dtype/shape except axis 0)
d0 = arrays[0].dtype
rest_shape = arrays[0].shape[1:]
assert all(a.dtype == d0 for a in arrays), "dtype mismatch"
assert all(a.shape[1:] == rest_shape for a in arrays), "shape mismatch (except axis 0)"

merged = np.concatenate(arrays, axis=0).astype(np.int32)  # change axis if you need
np.save(out, merged)
print(f"Saved: {out} with shape {merged.shape}, dtype {merged.dtype}")
