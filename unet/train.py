import sys
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
sys.path.append(os.path.abspath(os.path.join("..")))
from unet.utils import dataset, model
import numpy as np
import matplotlib.pyplot as plt

# Define the desired shape
target_shape_img = [128, 128, 3]
target_shape_mask = [128, 128, 1]

path1 = os.path.abspath(os.path.join("..", "segmentation_dataset_balanced", "train", "images"))
path2 = os.path.abspath(os.path.join("..", "segmentation_dataset_balanced", "train", "masks"))

img, mask = dataset.LoadData(path1, path2)

# Process data using apt helper function
X, y = dataset.PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2)

# QC the shape of output and classes in output dataset 
print("X Shape:", X.shape)
print("Y shape:", y.shape)
# There are 3 classes : background, PRENDA DE ARRIBA, PRENDA DE ABAJO
print(np.unique(y))

# Visualize the output
image_index = 0
fig, arr = plt.subplots(1, 2, figsize=(15, 15))
arr[0].imshow(X[image_index])
arr[0].set_title('Processed Image')
arr[1].imshow(y[image_index,:,:,0])
arr[1].set_title('Processed Masked Image ')

# Save a preview image
plt.savefig(os.path.join("..", "segmentation_dataset_balanced", "proccess", "preview.png"))

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=123)

unet = model.UNetCompiled(input_size=(128,128,3), n_filters=32, n_classes=3)

# Check the summary to better interpret how the output dimensions change in each layer
unet.summary()

unet.compile(optimizer=tf.keras.optimizers.Adam(),  # type: ignore
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # type: ignore
              metrics=['accuracy'])

results = unet.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid))

plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig(os.path.join("..", "segmentation_dataset_balanced", "proccess", "loss_plot.png"))

unet.save("equipo_8_modelo.keras")
