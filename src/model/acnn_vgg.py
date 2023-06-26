import tensorflow as tf
from tensorflow import keras
from src.layers.block import AttentionBlock


def _make_convolution_block(filters, num_conv_layers):
    block = keras.Sequential()
    for i in range(num_conv_layers):
        block.add(keras.layers.Conv2D(filters, (3, 3), padding='same', bias_initializer='glorot_uniform'))
        block.add(keras.layers.BatchNormalization())
        block.add(keras.layers.ReLU())

    return block


class AtentionVGG(keras.Model):
    def __init__(self, hidden_units, num_classes, use_pretrained=False):
        super(AtentionVGG, self).__init__()

        if use_pretrained:
            backbone = keras.applications.VGG16(include_top=False, weights="imagenet")
            self.block_1 = keras.Sequential(backbone.layers[1:3])
            self.block_2 = keras.Sequential(backbone.layers[4:6])
            self.block_3 = keras.Sequential(backbone.layers[7:10])
            self.block_4 = keras.Sequential(backbone.layers[11:14])
            self.block_5 = keras.Sequential(backbone.layers[15:18])
        else:
            self.block_1 = _make_convolution_block(64, 2)
            self.block_2 = _make_convolution_block(128, 2)
            self.block_3 = _make_convolution_block(256, 3)
            self.block_4 = _make_convolution_block(512, 3)
            self.block_5 = _make_convolution_block(512, 3)
        self.max_pool = keras.layers.MaxPooling2D()
        self.global_avg_pool = keras.layers.GlobalAveragePooling2D()
        self.dropout = keras.layers.Dropout(0.2)
        self.dense = keras.layers.Dense(hidden_units, activation='relu')
        self.output_fc = keras.layers.Dense(num_classes)
        self.attn_1 = AttentionBlock(256, (28, 28))
        self.attn_2 = AttentionBlock(256, (14, 14))

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.max_pool(x)
        x = self.block_2(x)
        x = self.max_pool(x)
        block_3 = self.block_3(x)
        pool_3 = self.max_pool(block_3)
        block_4 = self.block_4(pool_3)
        pool_4 = self.max_pool(block_4)
        block_5 = self.block_5(pool_4)
        global_feature = self.max_pool(block_5)

        global_vector = self.global_avg_pool(global_feature)
        local_1, _ = self.attn_1(pool_3, global_feature)
        local_2, _ = self.attn_2(pool_4, global_feature)
        x = tf.concat((global_vector, local_1, local_2), axis=1)
        x = self.dropout(x)
        x = self.dense(x)
        logits = self.output_fc(x)

        return logits

    def predict(self, inputs, as_int=False):
        x = self.block_1(inputs)
        x = self.max_pool(x)
        x = self.block_2(x)
        x = self.max_pool(x)
        block_3 = self.block_3(x)
        pool_3 = self.max_pool(block_3)
        block_4 = self.block_4(pool_3)
        pool_4 = self.max_pool(block_4)
        block_5 = self.block_5(pool_4)
        global_feature = self.max_pool(block_5)

        global_vector = self.global_avg_pool(global_feature)
        local_1, attn_weights_1 = self.attn_1(pool_3, global_feature)
        local_2, attn_weights_2 = self.attn_2(pool_4, global_feature)
        x = tf.concat((global_vector, local_1, local_2), axis=1)
        x = self.dropout(x)
        x = self.dense(x)
        logits = self.output_fc(x)

        if as_int:
            pred_labels = tf.cast(logits > 0.0, dtype=tf.int32)
        else:
            pred_labels = tf.cast(logits > 0.0, dtype=tf.float32)

        return pred_labels, attn_weights_1, attn_weights_2
