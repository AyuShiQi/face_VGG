import paddle

from VGG import VGG
from data_preprocess import data_loader

DATADIR = 'train'
DATADIR2 = 'val'
DATADIR3 = 'test'


def train_pm(model, optimizer, loss_fct, EPOCH_NUM):
    """
    训练模型
    :param model: 模型
    :param optimizer: 优化器
    :param loss_fct: 损失函数
    :param EPOCH_NUM: 迭代次数
    :return:
    """
    # 开启0号GPU训练(暂时不启用)
    use_gpu = False
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

    print('start training ... ')
    model.train()
    # 定义数据读取器，训练数据读取器和验证数据读取器
    train_loader = data_loader(DATADIR, batch_size=20, mode='train')
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data

            # 将图片和标签都转化为tensor型
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)

            # 运行模型前向计算，得到预测值
            logits = model(img)
            # 计算输入input和标签label间的交叉熵损失
            avg_loss = loss_fct(logits, label)

            if batch_id % 200 == 0:
                print("epoch: {}, batch_id: {}, loss is: {:.4f}".format(epoch, batch_id, float(avg_loss.numpy())))

            # 反向传播，更新权重，清除梯度
            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        # 保存模型
        paddle.save(model.state_dict(), 'vgg.pdparams')
        paddle.save(optimizer.state_dict(), 'vgg.pdopt')


model = VGG(num_class=7)  # 创建模型
loss_fct = paddle.nn.CrossEntropyLoss()  # 结合了LogSoftmax和NLLLoss的OP计算，可用于训练一个n类分类器。
# learning_rate为学习率，用于参数更新的计算。momentum为动量因子
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
EPOCH_NUM = 100  # 训练进行100次迭代
# 启动训练过程
train_pm(model, opt, loss_fct, EPOCH_NUM)
