import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram(grayfig):
    x = grayfig.shape[0]
    y = grayfig.shape[1]
    ret = np.zeros(256)
    for i in range(x):
        for j in range(y):
            ret[grayfig[i][j]] += 1
    return ret

img = 'src/car.jpg'
imgarray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)  # 打开图片,并把图片转换为np矩阵
ret = histogram(imgarray)
# plt.title("Histogram")
# plt.plot(ret)
# plt.axis([0, 255, 0, np.max(ret) * 1.05])
# plt.xlabel("Gray Level")
# plt.ylabel("Frequence")
plt.figure(figsize=(300,100), dpi=10)
plt.xlabel('Gray level')
plt.ylabel('Frequence')
plt.grid()
plt.bar(range(256), ret)
plt.show()
