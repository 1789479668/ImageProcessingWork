import cv2
import matplotlib.pyplot as plt
import numpy as np
def histogarm(img):
    # image
    imgarray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)#打开图片,并把图片转换为np矩阵
    # create a histogram of the image
    hist = np.zeros(256)
    count = 0
    for row in imgarray:
        for gray_data in row:
            hist[gray_data] += 1
            count += 1
    hist = hist / count
    # create the plot
    plt.title("Histogram")
    plt.plot(hist)
    plt.axis([0, 255, 0, np.max(hist) * 1.05])
    plt.xlabel("Gray Level")
    plt.ylabel("Frequence")
    plt.show()

img_path = 'src/Couple.bmp'
histogarm(img_path)