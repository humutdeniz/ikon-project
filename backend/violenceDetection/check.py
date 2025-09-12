# sanity_check.py
import pathlib
from detection import makeDataset, numFrames, targetSize

def assertDir(p: pathlib.Path):
    if not p.exists():
        raise SystemExit(f"Missing: {p}")

def main():
    root = pathlib.Path("data")
    for split in ["train", "val"]:
        for cls in ["nonfight", "fight"]:
            assertDir(root / split / cls)
    print("✅ Folder layout looks good.")

    ds = makeDataset("train", True)
    for xb, yb in ds.take(1):
        print(f"batch: {xb.shape} labels: {yb.numpy()}")
        # Expect (B, {numFrames}, {targetSize}, {targetSize}, 3)
    print("✅ Dataset pipeline produces batches. You’re good to train.")

if __name__ == "__main__":
    main()
