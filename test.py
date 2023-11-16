import numpy as np
import paddle

from VGG import VGG
from data_preprocess import data_loader
from main import DATADIR3, loss_fct


@paddle.no_grad()
# 定义评估函数
def evaluation(model, loss_fct):
    print('start evaluation .......')
    model.eval()
    eval_loader = data_loader(DATADIR3,
                              batch_size=20, mode='eval')

    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(eval_loader()):
        x_data, y_data = data
        # 将图片和标签都转化为tensor型
        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)

        # 计算预测和精度
        logits = model(img)
        acc = paddle.metric.accuracy(logits, label)
        avg_loss = loss_fct(logits, label)

        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))
    # 求平均精度
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()
    model.train()
    print('loss={:.4f}, acc={:.4f}'.format(avg_loss_val_mean, acc_val_mean))


model = VGG(num_class=7)  # 创建模型
# 开启0号GPU预估
use_gpu = False
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
# 加载模型参数
params_file_path = './models/vgg.pdparams'
model_state_dict = paddle.load(params_file_path)
model.load_dict(model_state_dict)
# 调用验证
evaluation(model, loss_fct)
