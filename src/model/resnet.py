from src.layers.block import *


def _make_stage(filters, s, num_identity_blocks):
    stage = keras.Sequential()
    stage.add(ConvolutionBlock(filters=filters, s=s))
    for i in range(num_identity_blocks):
        stage.add(IdentityBlock(filters=filters))

    return stage


class ResNet50(keras.Model):
    def __init__(self, hidden_units, num_classes):
        super(ResNet50, self).__init__()

        self.zero_padding = keras.layers.ZeroPadding2D(padding=(3, 3))

        self.conv2d = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))
        self.bn = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')
        self.max_pooling = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))

        self.global_avg_pooling = keras.layers.GlobalAveragePooling2D()
        self.dense = keras.layers.Dense(hidden_units, activation='relu')
        self.dropout = keras.layers.Dropout(0.3)
        self.fc = keras.layers.Dense(num_classes)

        self.stage_1 = _make_stage(filters=(64, 256), s=1, num_identity_blocks=2)
        self.stage_2 = _make_stage(filters=(128, 512), s=2, num_identity_blocks=3)
        self.stage_3 = _make_stage(filters=(256, 1024), s=2, num_identity_blocks=5)
        self.stage_4 = _make_stage(filters=(512, 2048), s=2, num_identity_blocks=2)

    def call(self, inputs):
        x = self.zero_padding(inputs)

        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)

        x = self.global_avg_pooling(x)
        x = self.dropout(x)
        x = self.dense(x)
        logits = self.fc(x)

        return logits

    def predict(self, inputs, as_int=False):
        logits = self(inputs, training=False)
        if as_int:
            pred_labels = tf.cast(logits > 0.0, dtype=tf.int32)
        else:
            pred_labels = tf.cast(logits > 0.0, dtype=tf.float32)
        return pred_labels
