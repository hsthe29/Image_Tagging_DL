import tensorflow as tf
from tensorflow import keras
from src.layers.block import IdentityBlock, LargeKernelConvolution, ConvolutionBlock


class LKResidualNetwork(keras.Model):
    def __init__(self, hidden_units, num_classes):
        super(LKResidualNetwork, self).__init__()

        self.zero_padding = keras.layers.ZeroPadding2D(padding=(3, 3))
        self.conv2d = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(1, 1), bias_initializer='glorot_uniform')
        self.bn = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')
        self.max_pooling = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))  # 112, 112, 64

        self.lk_1 = LargeKernelConvolution(filters=(64, 256), s=2)  # 56, 56, 256
        self.lk_2 = LargeKernelConvolution(filters=(128, 512), s=2)  # 28, 28, 512
        self.lk_3 = LargeKernelConvolution(filters=(192, 768), s=2)  # 14, 14, 768
        self.conv_block = ConvolutionBlock(filters=(256, 1024), s=2)  # 7, 7, 1024
        self.identity_block = IdentityBlock(filters=(256, 1024))

        self.global_avg_pooling = keras.layers.GlobalAveragePooling2D()
        self.dropout = keras.layers.Dropout(0.3)
        self.dense = keras.layers.Dense(hidden_units, activation='relu')
        self.dropout_fc = keras.layers.Dropout(0.2)
        self.fc = keras.layers.Dense(num_classes)

    def call(self, inputs):
        x = self.zero_padding(inputs)
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x = self.lk_1(x)
        x = self.lk_2(x)
        x = self.lk_3(x)
        x = self.conv_block(x)
        x = self.identity_block(x)

        x = self.global_avg_pooling(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.dropout_fc(x)
        logits = self.fc(x)

        return logits

    def predict(self, inputs, as_int=False):
        logits = self(inputs, training=False)

        if as_int:
            pred_labels = tf.cast(logits > 0.0, dtype=tf.int32)
        else:
            pred_labels = tf.cast(logits > 0.0, dtype=tf.float32)

        return pred_labels
