from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

imgpath = 'src/Miss.bmp'

img = Image.open(imgpath)

img.show()