import tensorflow as tf
from tensorflow import keras


class IdentityBlock(keras.layers.Layer):
    def __init__(self, filters):
        super(IdentityBlock, self).__init__()

        f1, f2 = filters

        self.conv2d_a = keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid')
        self.bn_a = keras.layers.BatchNormalization()

        self.conv2d_b = keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn_b = keras.layers.BatchNormalization()

        self.conv2d_c = keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid')
        self.bn_c = keras.layers.BatchNormalization()

        self.add = keras.layers.Add()
        self.relu = keras.layers.Activation('relu')

    def call(self, inputs):
        x_shortcut = inputs
        x = inputs

        x = self.conv2d_a(x)
        x = self.bn_a(x)
        x = self.relu(x)

        x = self.conv2d_b(x)
        x = self.bn_b(x)
        x = self.relu(x)

        x = self.conv2d_c(x)
        x = self.bn_c(x)

        x = self.add([x, x_shortcut])
        x = self.relu(x)

        return x


class ConvolutionBlock(keras.layers.Layer):
    def __init__(self, filters, s):
        super(ConvolutionBlock, self).__init__()

        f1, f2 = filters

        self.conv2d_1a = keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid')
        self.bn_1a = keras.layers.BatchNormalization()

        self.conv2d_1b = keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn_1b = keras.layers.BatchNormalization()

        self.conv2d_1c = keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid')
        self.bn_1c = keras.layers.BatchNormalization()

        self.conv2d_2 = keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid')
        self.bn_2 = keras.layers.BatchNormalization()

        self.add = keras.layers.Add()
        self.relu = keras.layers.Activation('relu')

    def call(self, inputs):
        x_shortcut = inputs
        x = inputs

        x = self.conv2d_1a(x)
        x = self.bn_1a(x)
        x = self.relu(x)

        x = self.conv2d_1b(x)
        x = self.bn_1b(x)
        x = self.relu(x)

        x = self.conv2d_1c(x)
        x = self.bn_1c(x)

        x_shortcut = self.conv2d_2(x_shortcut)
        x_shortcut = self.bn_2(x_shortcut)

        x = self.add([x, x_shortcut])
        x = self.relu(x)

        return x


class LargeKernelConvolution(keras.layers.Layer):
    def __init__(self, filters, s):
        super(LargeKernelConvolution, self).__init__()

        in_channels, out_channels = filters
        self.in_conv = keras.layers.Conv2D(
            in_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            bias_initializer='glorot_uniform'
        )
        self.dw_conv = keras.layers.DepthwiseConv2D(
            kernel_size=(5, 5),
            padding='same',
            bias_initializer='glorot_uniform'
        )
        self.zero_padding = keras.layers.ZeroPadding2D(padding=(9, 9))
        self.dwd_conv = keras.layers.DepthwiseConv2D(
            kernel_size=(7, 7),
            padding='valid',
            dilation_rate=3,
            bias_initializer='glorot_uniform'
        )
        self.pw_conv = keras.layers.Conv2D(
            in_channels,
            kernel_size=(1, 1),
            bias_initializer='glorot_uniform'
        )

        self.out_conv2d = keras.layers.Conv2D(out_channels, kernel_size=(3, 3), strides=(s, s), padding='same',
                                              bias_initializer='glorot_uniform')
        self.in_bn = keras.layers.BatchNormalization()
        self.out_bn = keras.layers.BatchNormalization()
        self.mul = keras.layers.Multiply()
        self.relu = keras.layers.Activation('relu')

    def call(self, inputs):
        x = self.in_conv(inputs)
        x_shortcut = x
        x = self.in_bn(x)
        x = self.dw_conv(x)
        x = self.zero_padding(x)
        x = self.dwd_conv(x)

        attention_weight = tf.nn.sigmoid(self.pw_conv(x))
        x = self.mul([x_shortcut, attention_weight])
        x = self.out_conv2d(x)
        x = self.out_bn(x)
        x = self.relu(x)

        return x


class AttentionBlock(keras.layers.Layer):
    def __init__(self, attn_features, up_sampling_size):
        super(AttentionBlock, self).__init__()
        self.up_sampling_size = up_sampling_size
        self.W_f = keras.layers.Conv2D(attn_features, kernel_size=1, padding='same', use_bias=False)
        self.W_g = keras.layers.Conv2D(attn_features, kernel_size=1, padding='same', use_bias=False)
        self.W = keras.layers.Conv2D(1, kernel_size=1, padding='same', bias_initializer='glorot_uniform')
        self.glob_avg = keras.layers.GlobalAveragePooling2D()

    def call(self, f, g):
        f_ = self.W_f(f)
        g_ = self.W_g(g)
        g_ = tf.image.resize(g_, self.up_sampling_size)

        c = self.W(tf.nn.relu(f_ + g_))

        attention_weights = tf.math.sigmoid(c)
        x = tf.math.multiply(attention_weights, f)
        x = self.glob_avg(x)
        return x, attention_weights
