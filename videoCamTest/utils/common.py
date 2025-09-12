
import os, time, torch, numpy as np

def setDeterministic(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensureDir(path: str):
    os.makedirs(path, exist_ok=True)

def nowSeconds():
    return time.time()

def normalizeRoiPolygon(polygon, w, h):
    # polygon is normalized [0..1]; convert to pixel coords
    return [(int(x*w), int(y*h)) for x,y in polygon]
