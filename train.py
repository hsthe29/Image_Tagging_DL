from src.data import *
from src.metrics import *
from src.losses import *
from src.utils import *
from src.model.base_resnet_50 import *
from src.model.resnet import *
from src.model.acnn_vgg import *
from src.model.thenet import *
import yaml

if __name__ == "__main__":
    with open("config.yml") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    labels = config['labels']
    NUM_CLASSES = len(labels)
    raw_data = get_data_from_csv(config['train_csv'])
    image_dir = config['train_dir']
    image_paths, labels = extract_path_and_label(image_dir, raw_data)

    train_raw, val_raw = get_dataset(image_paths, labels, ratio=config['train_ratio'])

    train_ds = train_raw.map(convert_path, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_raw.map(convert_path, num_parallel_calls=tf.data.AUTOTUNE)

    hidden_units = config['hidden_units']

    model = BaseResNet50V2(hidden_units, NUM_CLASSES)
    # model = ResNet50(hidden_units, NUM_CLASSES)
    # model = ACNNv1(hidden_units, NUM_CLASSES)
    # model = ACNNv2(hidden_units, NUM_CLASSES)

    model.compile(optimizer='adam',
                  loss=entropy_loss_with_logits, metrics=[accuracy_score, f1_score])

    print("Training: ")
    history = model.fit(
        train_ds,
        epochs=config['epochs'],
        validation_data=val_ds,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])

    model.save_weights('tmp/model_weights.tf')

    plot_history(history.history)

    # Evaluate model
