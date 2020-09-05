import tensorflow as tf
from scipy.stats import norm
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy
import random
import time
import pickle
import os, sys

# 加载模型
print("Loading model..")
network = tf.keras.models.load_model('my_new_model.h5')
network.trainable = False
print("Complete load model.")

'''正确样本'''
correct_image, correct_label = pickle.load(open('my_new_correct_1k.pkl', 'rb'))
correct_image = numpy.asarray(correct_image, dtype=numpy.float32).reshape((-1, 28, 28, 1))
correct_label = numpy.asarray(correct_label, dtype=numpy.float32).reshape((-1, 10))
print(correct_image.shape, correct_label.shape)
print(correct_label[0])
loss, acc = network.evaluate(correct_image, correct_label)

'''攻击样本'''
attack_image, attack_label = pickle.load(open('my_new_white_attack_1k.pkl','rb'))
attack_image = numpy.asarray(attack_image, dtype=numpy.float32).reshape((-1, 28, 28, 1))
attack_label = numpy.asarray(attack_label, dtype=numpy.float32).reshape((-1, 10))
print(attack_image.shape, attack_label.shape)
print(attack_label[0])

# 训练集攻击成功率
loss, acc = network.evaluate(attack_image, attack_label)

t=0
idx = random.sample(list(range(1000)), 100)
for i in idx:
    x_attack = attack_image[i]
    x_attack = x_attack[numpy.newaxis, :]
    y_attack = attack_label[i]
    y_attack = y_attack[numpy.newaxis, :]

    loss, acc = network.evaluate(x_attack, y_attack)
    if acc == 1:
        t += 1
        num_attack = numpy.argmax(attack_label[i])
        num_correct = numpy.argmax(correct_label[i])

        plt.imshow(numpy.squeeze(attack_image[i]))
        plt.imsave("../sample/%d_attack_%d.jpg" % (i, num_attack), numpy.squeeze(attack_image[i]))

        plt.imshow(numpy.squeeze(correct_image[i]))
        plt.imsave("../sample/%d_correct_%d.jpg" % (i, num_correct), numpy.squeeze(correct_image[i]))

    if t == 10:
        break


