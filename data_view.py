import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# %matplotlib inline

# 选取 test文件夹 作为图片路径
DATADIR = 'test'

file0 = 'NE00127.jpg'
file1 = 'HA00453.jpg'
file2 = 'SU00224.jpg'

# 读取图片
img0 = Image.open(os.path.join(DATADIR, file0))  # NE00127.jpg
img0 = np.array(img0)
img1 = Image.open(os.path.join(DATADIR, file1))  # HA00453.jpg
img1 = np.array(img1)
img2 = Image.open(os.path.join(DATADIR, file2))  # SU00224.jpg
img2 = np.array(img2)
# 画出读取的图片
plt.figure(figsize=(16, 8))

f = plt.subplot(131)
f.set_title('0', fontsize=20)
plt.imshow(img0)

f = plt.subplot(132)
f.set_title('1', fontsize=20)
plt.imshow(img1)

f = plt.subplot(133)
f.set_title('2', fontsize=20)
plt.imshow(img2)
# plt展示出三个表情
plt.show()
