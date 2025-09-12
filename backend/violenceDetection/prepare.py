# prepare_rwf.py
import argparse, pathlib, random, shutil, os

def listVideos(rootDir):
    return [p for p in pathlib.Path(rootDir).rglob("*.mp4")]

def ensureDir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def pickSplit(items, k):
    random.shuffle(items)
    return set(items[:k])

def classDirNames():
    # normalize any of: Fight/NonFight, fight/nonfight, etc.
    return {"fight": ["fight", "Fight"], "nonfight": ["nonfight", "NonFight", "non-fight", "Non-fight"]}

def gatherByClass(base):
    byClass = {"fight": [], "nonfight": []}
    names = classDirNames()
    for cls, variants in names.items():
        for v in variants:
            d = pathlib.Path(base) / v
            if d.exists():
                byClass[cls] += [str(p) for p in d.glob("*.mp4")]
    return byClass

def writeLinks(files, outDir, copyFiles=False):
    ensureDir(outDir)
    for src in files:
        dst = pathlib.Path(outDir) / pathlib.Path(src).name
        if dst.exists(): 
            continue
        if copyFiles:
            shutil.copy2(src, dst)
        else:
            try:
                os.symlink(os.path.abspath(src), dst)
            except FileExistsError:
                pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to RWF-2000 root (folder that contains train/val or raw videos)")
    parser.add_argument("--target", default="data", help="Output dataset root (default: data)")
    parser.add_argument("--valRatio", type=float, default=0.15)
    parser.add_argument("--testRatio", type=float, default=0.15)
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    random.seed(args.seed)

    src = pathlib.Path(args.source)
    tgt = pathlib.Path(args.target)

    hasTrainVal = (src / "train").exists() and (src / "val").exists()

    if hasTrainVal:
        # Use given train/val; carve test from train (disjoint).
        trainByClass = gatherByClass(src / "train")
        valByClass = gatherByClass(src / "val")

        for cls in ["fight", "nonfight"]:
            nTrain = len(trainByClass[cls])
            kTest = max(1, int(nTrain * args.testRatio))
            testSet = pickSplit(list(trainByClass[cls]), kTest)
            trainRemain = [p for p in trainByClass[cls] if p not in testSet]

            writeLinks(trainRemain, tgt / "train" / cls, copyFiles=args.copy)
            writeLinks(valByClass[cls], tgt / "val" / cls, copyFiles=args.copy)
            writeLinks(list(testSet), tgt / "test" / cls, copyFiles=args.copy)

    else:
        # No splits provided → do 70/15/15 over everything we find.
        byClass = {"fight": [], "nonfight": []}
        for cls, variants in classDirNames().items():
            for v in variants:
                d = src / v
                if d.exists():
                    byClass[cls] += [str(p) for p in d.glob("*.mp4")]
        # If still empty, just scan all mp4s and guess class from filename (contains 'fight')
        if not byClass["fight"] and not byClass["nonfight"]:
            allVideos = [str(p) for p in src.rglob("*.mp4")]
            for p in allVideos:
                (byClass["fight"] if "fight" in p.lower() else byClass["nonfight"]).append(p)

        for cls in ["fight", "nonfight"]:
            vids = list(byClass[cls])
            random.shuffle(vids)
            n = len(vids)
            nVal = int(n * args.valRatio)
            nTest = int(n * args.testRatio)
            valSet = set(vids[:nVal])
            testSet = set(vids[nVal:nVal+nTest])
            trainSet = [p for p in vids if p not in valSet and p not in testSet]

            writeLinks(trainSet, tgt / "train" / cls, copyFiles=args.copy)
            writeLinks(list(valSet), tgt / "val" / cls, copyFiles=args.copy)
            writeLinks(list(testSet), tgt / "test" / cls, copyFiles=args.copy)

    print(f"✅ Prepared splits under: {tgt.resolve()}")

if __name__ == "__main__":
    main()
