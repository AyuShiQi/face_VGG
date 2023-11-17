import cv2
import numpy as np
from VGG import VGG
import paddle
import paddle.inference as paddle_infer
from data_process import transform_img

model_file_path = './models/vgg.pdmodel'
params_file_path = './models/vgg.pdiparams'

img_path = './test/AN00002.jpg'


def excute_detect (path):
    config = paddle_infer.Config(model_file_path, params_file_path)
    predictor = paddle_infer.create_predictor(config)

    print(predictor)
    img = cv2.imread(path)
    # print('开始打印', img.shape)

    # param_dict = paddle.load(params_file_path)
    # model.load_dict(param_dict)
    # 灌入数据
    # model.eval()
    tensor_img = transform_img(img)
    # print('tensor', tensor_img.shape)
    tensor_img = np.expand_dims(tensor_img, 0)
    # print('tensor', tensor_img.shape)

    # results = model(paddle.to_tensor(tensor_img))
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    print(input_names, input_handle)
    fake_input = np.random.randn(64, 3, 3, 3).astype("float32")
    input_handle.reshape([64])
    input_handle.copy_from_cpu(fake_input)
    # print('results', results, results.numpy())
    # 取概率最大的标签作为预测输出
    res = predictor.run()
    print(res)


excute_detect(img_path)
