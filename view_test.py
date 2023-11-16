import cv2
import numpy as np
from VGG import VGG
import paddle
from data_process import transform_img

model = VGG(num_class=7)
params_file_path = './models/vgg.pdparams'

img_path = './test/AN00002.jpg'
img = cv2.imread(img_path)

print('开始打印', img.shape)

param_dict = paddle.load(params_file_path)
model.load_dict(param_dict)
# 灌入数据
model.eval()
tensor_img = transform_img(img)
print('tensor', tensor_img.shape)
tensor_img = np.expand_dims(tensor_img, 0)
print('tensor', tensor_img.shape)

results = model(paddle.to_tensor(tensor_img))
print('results', results, results.numpy())
# 取概率最大的标签作为预测输出
lab = np.argsort(results.numpy())
print('lab', lab)
print("这次预测图片名称是：%s" % img_path)
if img_path[7: 9] == 'SA':
    true_lab = 'sad'
elif img_path[7: 9] == 'DI':
    true_lab = 'disgust'
elif img_path[7: 9] == 'HA':
    true_lab = 'happy'
elif img_path[7: 9] == 'FE':
    true_lab = 'fear'
elif img_path[7: 9] == 'SU':
    true_lab = 'surprise'
elif img_path[7: 9] == 'NE':
    true_lab = 'neutral'
elif img_path[7: 9] == 'AN':
    true_lab = 'angry'
else:
    raise 'Not excepted file name'

print("这次图片属于%s表情" % true_lab)
tap = lab[0][-1]
print("这次预测结果是：")
if tap == 0:
    print('sad')
elif tap == 1:
    print('disgust')
elif tap == 2:
    print('happy')
elif tap == 3:
    print('fear')
elif tap == 4:
    print('surprise')
elif tap == 5:
    print('neutral')
elif tap == 6:
    print('angry')
else:
    raise 'Not excepted file name'
