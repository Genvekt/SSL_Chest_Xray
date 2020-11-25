from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

def showInRow(list_of_images, titles = None, 
              disable_ticks = False, vertical=False,
              tensor=False):
    tensor_to_img = transforms.ToPILImage()
    to_gray = transforms.Grayscale()
    count = len(list_of_images)
    for idx in range(count):
        if vertical:
            subplot = plt.subplot(count, 1, idx+1)
        else:
            subplot = plt.subplot(1, count, idx+1)
        if titles is not None:
            subplot.set_title(titles[idx])
        img = list_of_images[idx]
        if tensor:
#             img = tensor_to_img(img)
#             img = to_gray(img)
              img = img.permute(1, 2, 0)
        
        if len(img.shape) >2:
            img = img[:,:,0]
            
        cmap = 'gray' if (len(img.shape) == 2 or img.shape[2] == 1) else None
        subplot.imshow(img, cmap=cmap)
        if disable_ticks:
            plt.xticks([]), plt.yticks([])
    plt.show()