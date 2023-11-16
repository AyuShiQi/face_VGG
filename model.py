import paddle
from paddle.static import InputSpec

from VGG import VGG

model = VGG(num_class=7)
model_state_dict = paddle.load('./models/vgg.pdparams')
model.set_state_dict(model_state_dict)

net = paddle.jit.to_static(model, input_spec=[InputSpec(shape=[64, 3, 3, 3])])
paddle.jit.save(net, './models/vgg')

