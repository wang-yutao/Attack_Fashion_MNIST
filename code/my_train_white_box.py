import os
import numpy
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, losses, metrics, datasets
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略低级别警告
#
# tf.random.set_seed(2019)
# numpy.random.seed(2019)

print("Loading model..")
network = tf.keras.models.load_model('my_model.h5')
network.trainable = False
print("Complete load model.")

# 梯度接口
def grad_op(input_image, input_label):
    loss_object = losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = network(input_image)
        # prediction = tf.nn.softmax(prediction)
        loss = loss_object(input_label, prediction)

    gradient = tape.gradient(loss, input_image)
    # gradient = tf.sign(gradient)
    return gradient

def eval_op(input_image, input_label):
    loss, acc = network.evaluate(input_image, input_label)
    return loss, acc

def infer_op(input_image):
    prediction = network.predict(input_image)
    return prediction

correct_image, correct_label = pickle.load(open('my_correct_train_1k.pkl','rb'))
correct_image = numpy.asarray(correct_image, dtype=numpy.float32).reshape((-1, 28, 28, 1))
correct_label = numpy.asarray(correct_label, dtype=numpy.float32).reshape((-1, 10))
print(correct_image.shape, correct_label.shape)
# 将原正确类别转换为目标类别
attack_label = numpy.delete(correct_label, 9, axis=1)
attack_label = numpy.column_stack((correct_label[:, 9], attack_label))
attack_image = correct_image
# print(correct_image[0])
eval_op(correct_image, correct_label)
eval_op(correct_image, attack_label)

ratio = 0.01
for i in range(1000):
    x_0 = correct_image[i]  # (28, 28, 1)
    # plt.imshow(tf.squeeze(x_0))
    # plt.show()
    x_0 = norm.rvs(loc=x_0, scale=0.01)
    x_0 = tf.clip_by_value(x_0, 0, 1)
    x_0 = tf.expand_dims(x_0, axis=0)
    y_0 = attack_label[i]  # (10,)
    # print(y_0)
    # y_0 = norm.rvs(loc=y_0, scale=0.1)
    # y_0 = tf.clip_by_value(y_0, 0, 1)

    for j in range(1000):
        x_0 = tf.convert_to_tensor(x_0)
        y_0 = tf.convert_to_tensor(y_0)
        grad = grad_op(x_0, y_0)

        # print(grad)
        x_0 += ratio * grad + norm.rvs(scale=0.01)
        # noise = tf.clip_by_value(noise, -ratio, ratio)
        x_0 = tf.clip_by_value(x_0, 0, 1)

        y_hat = infer_op(x_0)
        # print(y_hat)
        y_hat = numpy.array(y_hat)  # (1, 1, 10)
        y_hat = numpy.squeeze(y_hat)  # (10,)
        # print(y_hat.shape)
        # print(y_0.shape)

        argmaxy_hat = numpy.argmax(y_hat)
        argmaxy_0 = numpy.argmax(y_0)

        # plt.imshow(tf.squeeze(x_0))
        # plt.show()
        if argmaxy_0 == argmaxy_hat:
            break

    attack_image[i] = x_0[0]
    y_0 = numpy.expand_dims(y_0, axis=0) # (1, 10)
    loss, acc = eval_op(x_0, y_0)
    # dis = numpy.linalg.norm(x_0 - x_1)
    print(i, '  ', j, '  ', acc, '  ', loss)

attack_image = numpy.asarray(attack_image, dtype=numpy.float32).reshape((-1, 1, 28, 28))
attack_label = numpy.asarray(attack_label, dtype=numpy.float32).reshape((-1, 1, 10))
with open("my_train_white_attack_1k.pkl", "wb") as f:
    pickle.dump([attack_image, attack_label], f)