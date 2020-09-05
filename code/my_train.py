import os
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, losses, metrics, datasets
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略低级别警告

# 预处理函数
def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255
    x = tf.expand_dims(x, axis=-1) # 由于卷积层维度为[None, 28, 28, 1]，故在axis=3扩展一维
    y = tf.one_hot(y, depth=10)
    return x, y

# 加载数据
(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape, y.shape, x.min(), x.max()) # 显示加载进来的数据的维度信息，便于理解

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(100)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(10000)

# 创建网络模型并装配模型
network = Sequential([
    layers.Conv2D(6, kernel_size=5, strides=1, padding='SAME', activation='relu',input_shape =(28,28,1)),
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
# network.build(input_shape=[None, 28, 28, 1])
# network.summary()
#
# # 设置回调功能
# filepath = 'my_model_1.h5' # 保存模型地址
# saved_model = tf.keras.callbacks.ModelCheckpoint(filepath, verbose = 1) # 回调保存模型功能
# tensorboard = tf.keras.callbacks.TensorBoard(log_dir = 'log') # 回调可视化数据功能
#
# # 执行训练与验证
# history = network.fit(train_db, epochs = 10, validation_data = test_db,validation_freq = 1, callbacks = [saved_model, tensorboard])
#
# # 显示训练与验证相关数据统计
# history.history

# 加载训练好的模型并进行测试
del network
print("Loading model..")
network = tf.keras.models.load_model('my_model_1.h5')
network.summary()
print("Complete load model.")
loss, acc = network.evaluate(test_db)
print("Test accuracy: %g%%" % (acc))
