import os
import numpy
import random
import pickle
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, losses, metrics, datasets
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略低级别警告

# 预处理函数
def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255# 将MNIST数据映射到[0，1]
    x = tf.expand_dims(x, axis=-1) # 由于卷积层维度为[None, 28, 28, 1]，故在axis=3扩展一维
    y = tf.one_hot(y, depth=10)
    return x, y

# 加载数据
(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape, y.shape, x.min(), x.max()) # 显示加载进来的数据的维度信息，便于理解

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess)

# 加载训练好的模型并进行测试
print("Loading model..")
network = tf.keras.models.load_model('my_model.h5')
print("Complete load model.")
# network.compile(optimizer=optimizers.Adam(lr=0.01),
#                 loss=losses.CategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy'])
# network.build(input_shape=[None, 28, 28, 1])

# 筛选测试集正确样本
acc = 0
correct_image = []
correct_label = []
for step, (x, y) in enumerate(train_db):
    # print(step)
    x = tf.expand_dims(x, axis=0)
    # print(x.shape, y.shape)
    y_hat = network.predict(x)
    if (numpy.argmax(y_hat) == numpy.argmax(y)):
        correct_image.append(x)
        correct_label.append(y)
        acc += 1

    if step == 2000:
        break

print(step)
acc /= step
print("[*] Accuracy on test set: %.5f" % (acc))

# 随机抽取 1000 张图片作为攻击样本
_correct_image = []
_correct_label = []
_idx = random.sample(range(len(correct_image)), 1000)
for i in _idx:
    _correct_image.append(correct_image[i])
    _correct_label.append(correct_label[i])
_correct_image = numpy.asarray(_correct_image, dtype=numpy.float32).reshape((-1, 1, 28, 28))
_correct_label = numpy.asarray(_correct_label, dtype=numpy.float32).reshape((-1, 1, 10))
with open("my_correct_train_1k.pkl", "wb") as f:
    pickle.dump([_correct_image, _correct_label], f)