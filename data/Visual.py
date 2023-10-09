import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def histogram(img, thres_img, img_c, thres):

    plt.figure(figsize=(15, 15))

    plt.subplot(3, 2, 1)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.title('Original Image')

    plt.subplot(3, 2, 2)
    sns.histplot(img.ravel(), bins=np.arange(0, 256), color='orange', alpha=0.5)
    sns.histplot(img[:, :, 0].ravel(), bins=np.arange(0, 256), color='red', alpha=0.5)
    sns.histplot(img[:, :, 1].ravel(), bins=np.arange(0, 256), color='Green', alpha=0.5)
    sns.histplot(img[:, :, 2].ravel(), bins=np.arange(0, 256), color='Blue', alpha=0.5)
    plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.ylim(0, 0.3e6)
    plt.xlabel('Intensity value')
    plt.title('Color Histogram')

    plt.subplot(3, 2, 3)
    plt.imshow(img_c, cmap='gist_gray')
    plt.title('Complement Grayscale Image')

    plt.subplot(3, 2, 4)
    sns.histplot(img_c.ravel(), bins=np.arange(0, 256))
    plt.axvline(thres, c='red', linestyle="--")
    plt.ylim(0, 0.3e6)
    plt.xlabel('Intensity value')
    plt.title('Grayscale Complement Histogram')

    plt.subplot(3, 2, 5)
    plt.imshow(thres_img, cmap='gist_gray')
    plt.title('Thresholded Image')

    plt.subplot(3, 2, 6)
    sns.histplot(thres_img.ravel(), bins=np.arange(0, 256))
    plt.axvline(thres, c='red', linestyle="--")
    plt.ylim(0, 0.3e6)
    plt.xlabel('Intensity value')
    plt.title('Thresholded Histogram')

    plt.tight_layout()
    plt.show()
def thresholding(img, method='otsu'):
    # convert to grayscale complement image
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_c = 255 - grayscale_img
    thres, thres_img = 0, img_c.copy()
    if method == 'otsu':
        thres, thres_img = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'triangle':
        thres, thres_img = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    return thres, thres_img, img_c