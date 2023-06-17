import tensorflow as tf
from tensorflow import keras
from src.utils import load_pretrained


class ResNeXt(keras.Model):
    def __init__(self, hidden_units, num_classes, cardinality=32, base_width=4):
        super(ResNeXt, self).__init__()
        # Pretrained model for feature extraction
        self.pretrained = load_pretrained()
        # Fix pretrained model's weights
        self.pretrained.trainable = False
        self.global_avg_pool = keras.layers.GlobalAveragePooling2D()
        self.dense = keras.layers.Dense(hidden_units, activation='relu')
        self.output_fc = keras.layers.Dense(num_classes)
        self.cardinality = cardinality
        self.base_width = base_width

    def call(self, batch_input):
        feature_maps = self.pretrained(batch_input)
        flatten_vectors = self.global_avg_pool(feature_maps)
        logits = self.dense(flatten_vectors)
        logits = self.output_fc(logits)

        return logits

    def predict(self, batch_input, as_int=True):
        batch_shape = tf.shape(batch_input)
        if len(batch_shape) == 3:
            batch_input = batch_input[tf.newaxis]

        logits = self(batch_input)
        pred = logits > 0.0
        if as_int:
            return tf.cast(pred, dtype=tf.int32)
        return tf.cast(pred, dtype=tf.float32)
