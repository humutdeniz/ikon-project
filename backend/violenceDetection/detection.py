# 2_data_pipeline.py
import os, cv2, random, pathlib, numpy as np, tensorflow as tf
from typing import List, Tuple

random.seed(1337)
np.random.seed(1337)
tf.random.set_seed(1337)

dataRoot = pathlib.Path("data")
classes = ["nonfight", "fight"]  # map to 0/1
classToIdx = {c: i for i, c in enumerate(classes)}

sampleFps = 5
numFrames = 16
targetSize = 172
batchSize = 1

numWorkers = tf.data.AUTOTUNE


def listFiles(split: str) -> List[Tuple[str, int]]:
    splitDir = dataRoot / split
    items = []
    for cls in classes:
        for p in (splitDir / cls).glob("*.avi"):
            items.append((str(p), classToIdx[cls]))
    random.shuffle(items)
    return items


def readVideo(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # pick indices roughly at sampleFps
    step = max(int(round(fps / sampleFps)), 1)
    idxs = list(range(0, total, step))[:numFrames]

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(
            frame, (targetSize, targetSize), interpolation=cv2.INTER_LINEAR
        )
        frames.append(frame.astype("float32") / 255.0)  # [0,1]
    cap.release()

    if len(frames) == 0:
        # fallback: black clip
        frames = [
            np.zeros((targetSize, targetSize, 3), dtype=np.float32)
            for _ in range(numFrames)
        ]

    # pad or trim to fixed length
    if len(frames) < numFrames:
        last = frames[-1]
        frames += [last] * (numFrames - len(frames))
    else:
        frames = frames[:numFrames]

    return np.stack(frames, axis=0)  # (T,H,W,3)


def pyLoad(path, label):
    clip = tf.numpy_function(
        func=lambda p: readVideo(p.decode("utf-8")), inp=[path], Tout=tf.float32
    )
    clip.set_shape((numFrames, targetSize, targetSize, 3))
    return clip, tf.cast(label, tf.int32)


def makeDataset(split: str, training: bool) -> tf.data.Dataset:
    pairs = listFiles(split)
    paths = tf.constant([p for p, _ in pairs])
    labels = tf.constant([y for _, y in pairs], dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(pairs), reshuffle_each_iteration=True)
    ds = ds.map(pyLoad, num_parallel_calls=numWorkers)
    if training:
        ds = ds.map(
            lambda x, y: (tf.image.random_flip_left_right(x), y),
            num_parallel_calls=numWorkers,
        )
    ds = ds.batch(batchSize).prefetch(numWorkers)
    return ds


if __name__ == "__main__":
    trainDs = makeDataset("train", True)
    valDs = makeDataset("val", False)
    for x, y in trainDs.take(1):
        print(x.shape, y.shape)  # (B,T,224,224,3) (B,)
