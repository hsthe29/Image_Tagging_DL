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
    raw_data = get_data_from_csv(config['test_csv'])
    image_dir = config['test_dir']
    image_paths, labels = extract_path_and_label(image_dir, raw_data)

    test_raw = get_dataset(image_paths, labels, test=True)

    test_ds = test_raw.map(convert_path, num_parallel_calls=tf.data.AUTOTUNE)

    hidden_units = config['hidden_units']

    model = BaseResNet50V2(hidden_units, NUM_CLASSES)
    # model = ResNet50(hidden_units, NUM_CLASSES)
    # model = AttentionVGG(hidden_units, NUM_CLASSES)
    # model = AttentionResNet(hidden_units, NUM_CLASSES)

    model.load_weights('tmp/model_weights.tf')

    # for _, (test_imgs, test_labels) in enumerate(test_ds):
    #     pred_labels = model.predict(test_imgs, as_int=False)
    #     apply metrics between `test_labels` and `pred_labels`
    #