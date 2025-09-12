# env_check.py
import sys, pkgutil
import tensorflow as tf
print("python:", sys.version)
print("tf:", tf.__version__)
print("tf keras ok:", tf.keras.layers.Dense is not None)
print("has tf-keras shim:", pkgutil.find_loader("tf_keras") is not None)

try:
    from official.projects.movinet.modeling import movinet, movinet_model
    print("tf-models-official movinet import: OK")
except Exception as e:
    print("movinet import error:", repr(e))
