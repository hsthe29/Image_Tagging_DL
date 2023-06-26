# Deep Learning task: Image Tagging
Project of course Deep learning and its applications. SOICT - HUST

## Description

**Included models:**
1. [Base ResNet50](src/model/base_resnet_50.py): Using pretrained Resnet50V2 model
2. [ResNet50](src/model/resnet.py): Implementing model based on ResNet architecture
3. [AttentionVGG](src/model/acnn_vgg.py): Implementing model based on Visual Attention Network
4. [TheNet](src/model/thenet.py): Using large kernel in convolution block and residual mechanism

## Run project
1. Run `$ ./run_build.sh` or `$ bash run_build.sh`
2. Edit project's configuration in [config.yml](config.yml)
3. For training: `$ python train.py`
4. For testing: `$ python test.py`
