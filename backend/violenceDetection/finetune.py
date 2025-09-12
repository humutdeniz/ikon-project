# 3_train_movinet.py

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_async_compilation=false"

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


import tensorflow as tf
import tf_keras as keras

# make GPU allocation incremental (prevents big upfront grabs)
try:
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

# belt & suspenders: also ask Keras not to JIT-compile graphs
# (add this flag to BOTH compile() calls below)
jitCompile = False

from official.projects.movinet.modeling import movinet, movinet_model

from detection import makeDataset, numFrames, targetSize

numClasses = 2
modelId = "a0"  # fast & small
baseRes = 224

trainDs = makeDataset("train", True)
valDs = makeDataset("val", False)

# backbone + classifier (build with Kinetics-600 size to load pretrained weights)
backbone = movinet.Movinet(model_id=modelId)
backbone.trainable = False
model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
model.build([None, numFrames, baseRes, baseRes, 3])

# load pretrained checkpoint
ckptUrl = f"https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_{modelId}_base.tar.gz"
ckptArchive = keras.utils.get_file(f"movinet_{modelId}_base.tar.gz", ckptUrl)
import tarfile, pathlib

with tarfile.open(ckptArchive) as tar:
    tar.extractall(path=".")
ckptDir = pathlib.Path(f"movinet_{modelId}_base")
tf.train.Checkpoint(model=model).restore(
    tf.train.latest_checkpoint(str(ckptDir))
).expect_partial()


# rebuild a new classifier head for 2 classes
def buildClassifier(backbone, numClasses):
    clf = movinet_model.MovinetClassifier(backbone=backbone, num_classes=numClasses)
    clf.build([None, numFrames, targetSize, targetSize, 3])
    return clf


head = buildClassifier(backbone, numClasses)

head.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
    jit_compile=False,
)
warmupEpochs = 3
head.fit(
    trainDs,
    validation_data=valDs,
    epochs=warmupEpochs,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=2, restore_best_weights=True
        )
    ],
)

# fine-tune (unfreeze some/all backbone)
head.backbone.trainable = True
for layer in head.backbone.layers[:]:  # optionally re-freeze early layers for stability
    layer.trainable = True

head.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
    jit_compile=False,
)
epochs = 10
history = head.fit(
    trainDs,
    validation_data=valDs,
    epochs=epochs,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            "ckpts/best.keras", save_best_only=True, monitor="val_accuracy"
        )
    ],
)

# export SavedModel
tf.saved_model.save(head, "export/saved_model")
print("Saved to export/saved_model")
