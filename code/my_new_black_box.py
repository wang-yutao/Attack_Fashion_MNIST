import os
import numpy
import pickle
import random
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, losses, metrics, datasets
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略低级别警告

print("Loading model..")
network = tf.keras.models.load_model('my_new_model.h5')
print("Complete load model.")

def eval_op(input_image, input_label):
    loss, acc = network.evaluate(input_image, input_label)
    return loss, acc

def infer_op(input_image):
    prediction = network.predict(input_image)
    return prediction

correct_image, correct_label = pickle.load(open('my_new_correct_1k.pkl','rb'))
correct_image = numpy.asarray(correct_image, dtype=numpy.float32).reshape((-1, 28, 28, 1))
correct_label = numpy.asarray(correct_label, dtype=numpy.float32).reshape((-1, 10))
print(correct_image.shape, correct_label.shape)
print(correct_label[0])
# 将原正确类别转换为目标类别
attack_label = numpy.delete(correct_label, 9, axis=1)
attack_label = numpy.column_stack((correct_label[:, 9], attack_label))
attack_image = correct_image

eval_op(correct_image, correct_label)
eval_op(correct_image, attack_label)

for i in range(1000):
    x_0 = correct_image[i]  # (28, 28, 1)
    y_0 = attack_label[i]  # (10,)
    argmaxy_0 = numpy.argmax(y_0)
    # print(x_0.shape)
    # print(y_0.shape)
    for j in range(1000):

        x_1 = norm.rvs(loc=x_0, scale=0.4)
        x_max = numpy.max(x_1)
        x_min = numpy.min(x_1)
        x_1 = numpy.round((x_1 - x_min) / (x_max - x_min))

        x_1 = tf.expand_dims(x_1, axis=0)  # (1, 28, 28, 1)

        y_hat = infer_op(x_1)
        y_hat = tf.nn.softsign(y_hat)
        # y_max = numpy.max(y_hat)
        # y_min = numpy.min(y_hat)
        # y_hat = numpy.round((y_hat - y_min) / (y_max - y_min))
        # print(y_hat)

        y_hat = numpy.array(y_hat)  # (1, 1, 10)
        y_hat = numpy.squeeze(y_hat)  # (10,)
        # print(y_hat.shape)
        # print(y_0.shape)

        argmaxy_hat = numpy.argmax(y_hat)
        # print(argmaxy_hat)
        # print(argmaxy_0)

        if argmaxy_0 == argmaxy_hat:
            x_0 = x_1
            break

        u = random.uniform(-1, 1)
        if u > y_hat[argmaxy_0]:
            x_0 = x_1

    attack_image[i] = x_0[0]
    y_0 = tf.expand_dims(y_0, axis=0)  # (1, 10)
    acc, loss = eval_op(x_0, y_0)
    # dis = numpy.linalg.norm(x_0 - x_1)
    print(i, '  ', j, '  ', acc, '  ', loss)

attack_image = numpy.asarray(attack_image, dtype=numpy.float32).reshape((-1, 1, 28, 28))
attack_label = numpy.asarray(attack_label, dtype=numpy.float32).reshape((-1, 1, 10))
with open("my_new_black_attack_1k.pkl", "wb") as f:
    pickle.dump([attack_image, attack_label], f)

