import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Conv2D works?", hasattr(tf.keras.layers, "Conv2D"))
