# coding :utf-8
import tensorflow as tf


def ConvBlock(out_channels, kernel, stride, x):
    x = tf.keras.layers.Conv2D(
        filters=out_channels, kernel_size=kernel,
        strides=stride,
        data_format='channels_last', activation='linear',
        kernel_initializer='he_normal', use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


class EnvNetV2(object):
    def __init__(self, input_shape, class_num):
        self.input_shape = input_shape
        self.class_num = class_num

    def build(self, is_train=True):
        input_data = tf.keras.layers.Input(
            shape=self.input_shape,
            name="sound_input"
        )

        x = ConvBlock(32, (64, 1), (2, 1), input_data)
        x = ConvBlock(64, (16, 1), (2, 1), x)
        x = tf.keras.layers.MaxPool2D(pool_size=(64, 1))(x)
        x = tf.keras.layers.Permute((1, 3, 2))(x)
        x = ConvBlock(32, (8, 8), 1, x)
        x = ConvBlock(32, (8, 8), 1, x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 5))(x)
        x = ConvBlock(64, (4, 1), 1, x)
        x = ConvBlock(64, (4, 1), 1, x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(x)
        x = ConvBlock(128, (2, 1), 1, x)
        x = ConvBlock(128, (2, 1), 1, x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(x)
        x = ConvBlock(256, (2, 1), 1, x)
        x = ConvBlock(256, (2, 1), 1, x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        if is_train:
            x = tf.keras.layers.Dropout(0.5)(x)
        else:
            x = tf.keras.layers.Dropout(0.0)(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        if is_train:
            x = tf.keras.layers.Dropout(0.5)(x)
        else:
            x = tf.keras.layers.Dropout(0.0)(x)
        output = tf.keras.layers.Dense(
            self.class_num, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=[input_data], outputs=[output])
        print(self.model.summary())

    def train(self, train_X, train_Y, batch_size=4, epochs=10):
        """Example Train"""
        self.build()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.categorical_crossentropy
        )
        self.model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs)
