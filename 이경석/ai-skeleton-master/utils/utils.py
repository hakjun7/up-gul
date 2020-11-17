from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf


# Req. 2-2	세팅 값 저장
def save_config():
    pass


# Req. 4-1	이미지와 캡션 시각화
def visualize_img_caption():
    images = os.path.abspath("images")
    img_name = input()
    caption = input()
    img = mpimg.imread(os.path.join(images, img_name))
    plt.title(f'<start> {caption} <end>')
    plt.imshow(img)
    plt.show()
