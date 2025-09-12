#!/usr/bin/env python3
import sys, torch, os

def pack(inPath: str, outPath: str):
    ckpt = torch.load(inPath, map_location="cpu")
    # already flat?
    if any(k.startswith("backbone.") or k.startswith("head.") for k in ckpt.keys()):
        torch.save(ckpt, outPath); return

    # expected: {"backbone": {...}, "head": {...}}
    flat = {}
    if "backbone" in ckpt:
        for k, v in ckpt["backbone"].items():
            flat[f"backbone.{k}"] = v
    if "head" in ckpt:
        for k, v in ckpt["head"].items():
            # rename "fc.weight" -> "weight", "fc.bias" -> "bias"
            newK = k[3:] if k.startswith("fc.") else k
            flat[f"head.{newK}"] = v

    if not flat:
        raise RuntimeError("checkpoint structure not recognized")
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    torch.save(flat, outPath)
    print(f"wrote {outPath}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: packCheckpoint.py <in.pth> <out.pth>"); sys.exit(1)
    pack(sys.argv[1], sys.argv[2])
