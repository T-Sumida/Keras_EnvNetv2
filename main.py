# coding: utf-8
import numpy as np
import tensorflow as tf
from envnetv2 import EnvNetV2

if __name__ == "__main__":
    AUDIO_LENGTH = 44100
    SAMPLE_NUM = 20
    CLASS_NUM = 5

    model = EnvNetV2((AUDIO_LENGTH, 1, 1), CLASS_NUM)

    train_X = np.random.rand(SAMPLE_NUM, AUDIO_LENGTH, 1, 1)
    train_Y = np.random.randint(0, CLASS_NUM-1, (SAMPLE_NUM, 1))
    train_Y = tf.keras.utils.to_categorical(train_Y, num_classes=CLASS_NUM)

    model.train(train_X, train_Y)
