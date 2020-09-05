import os
import numpy
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, losses, metrics, datasets
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略低级别警告

# 预处理函数
def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0 # 将MNIST数据映射到[0，1]
    x = tf.expand_dims(x, axis=-1) # 由于卷积层维度为[None, 28, 28, 1]，故在axis=3扩展一维
    y = tf.one_hot(y, depth=10)
    return x, y

# 加载数据
(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape, y.shape, x.min(), x.max()) # 显示加载进来的数据的维度信息，便于理解

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(10000)

# 加载模型
print("Loading model..")
network = tf.keras.models.load_model('my_model.h5')
network.trainable = False
print("Complete load model.")

def eval_op(input_image, input_label):
    loss, acc = network.evaluate(input_image, input_label)
    return loss, acc

def infer_op(input_image):
    prediction = network.predict(input_image)
    return prediction

retrain_image, retrain_label = pickle.load(open('my_train_white_attack_1k.pkl','rb'))
retrain_image = numpy.asarray(retrain_image, dtype=numpy.float32).reshape((-1, 28, 28, 1))
retrain_label = numpy.asarray(retrain_label, dtype=numpy.float32).reshape((-1, 10))
print(retrain_image.shape, retrain_label.shape)

# 训练集攻击成功率
eval_op(retrain_image, retrain_label)

retrain_x = []
retrain_y = []
count = 0
for i in range(1000):
    x_attack = retrain_image[i]
    y_attack = retrain_label[i]

    loss, acc = network.evaluate(x_attack[numpy.newaxis,:], y_attack[numpy.newaxis,:])
    if acc == 1:
        retrain_x.append(x_attack)
        retrain_y.append(y_attack)
        count += 1
    
retrain_x = numpy.array(retrain_x)
retrain_y = numpy.array(retrain_y)
print(count)
print(retrain_x.shape, retrain_y.shape, retrain_x.min(), retrain_y.max())
retrain_db = tf.data.Dataset.from_tensor_slices((retrain_x, retrain_y))

# 合并数据集
newtrain_db = train_db.concatenate(retrain_db)
newtrain_db = newtrain_db.shuffle(1000).batch(100)

# 创建网络模型并装配模型
del network
network = Sequential([
    layers.Conv2D(6, kernel_size=5, strides=1, padding='SAME', activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(pool_size=2, strides=2),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation=None)
])
network.compile(optimizer=optimizers.Adam(lr=0.001),
                loss=losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

# 显示网络结构信息
network.build(input_shape=[None, 28, 28, 1])
network.summary()

# 设置回调功能
filepath = 'my_new_model.h5' # 保存模型地址
saved_model = tf.keras.callbacks.ModelCheckpoint(filepath, verbose = 1) # 回调保存模型功能
tensorboard = tf.keras.callbacks.TensorBoard(log_dir = 'log') # 回调可视化数据功能

# 执行训练与验证
history = network.fit(newtrain_db, epochs = 10, callbacks = [saved_model, tensorboard])

# 显示训练与验证相关数据统计
history.history

# 加载训练好的模型并进行测试
del network
print("Loading model..")
network = tf.keras.models.load_model('my_new_model.h5')
print("Complete load model.")
loss, acc = network.evaluate(test_db)
print("Test accuracy: %g%%" % (acc))
