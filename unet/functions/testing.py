import os
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import unet.utils.dataset as dataset

""" Load Train Set and view some examples """
# Call the apt function
path1 = "segmentation_dataset/train/images"
path2 = "segmentation_dataset/train/masks"

img, mask = dataset.LoadData (path1, path2)

# View an example of image and corresponding mask 
show_images = 1
for i in range(show_images):
    img_view  = imageio.imread(path1 + img[i])
    mask_view = imageio.imread(path2 + mask[i])
    print(img_view.shape)
    print(mask_view.shape)
    fig, arr = plt.subplots(1, 2, figsize=(15, 15))
    arr[0].imshow(img_view)
    arr[0].set_title('Image '+ str(i))
    arr[1].imshow(mask_view)
    arr[1].set_title('Masked Image '+ str(i))
