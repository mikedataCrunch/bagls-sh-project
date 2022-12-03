from tensorflow.keras.applications import (
    InceptionResNetV2, ResNet50, InceptionV3, DenseNet121
)
import PATHS

nb_configs = {
    'validation_split' : 0.05,
    'rescale' : 1./255,
    'width_shift_range' : 0.1,
    'height_shift_range' : 0.1,
    'shear_range' : 0.1,
    'zoom_range' : 0.1,
    'fill_mode' : 'nearest',
    'horizontal_flip': True,
    'rotation_range': 30,  
#     'train_dir' : '../sample-dataset/train',
#     'test_dir' : '../sample-dataset/test',
    'train_dir' : '../dataset/train',
    'test_dir' : '../dataset/test',
    'batch_size' : 128,
    'class_names' : ["healthy", "unhealthy"],
    'interpol' : "bilinear",
    'cmap' : "rgb",
    'label_mode' : "categorical",
    'labels' : "inferred",
    'image_size' : (224, 224),
    'dropout_rate': 0.2,
    'thresh' : 0.5,
    'epochs': 10,
}

pt_configs = {
    "include_top": False,
    "input_shape": (*nb_configs["image_size"], 3),
    "pooling" : "avg",
}

pt_models = {
    "ResNet50" : ResNet50,
    "InceptionV3" : InceptionV3,
    "DenseNet121" : DenseNet121,
    "InceptionResNetV2" : InceptionResNetV2,
}

pt_weights = {
    "ResNet50": PATHS.resnet50_weights,
    "InceptionV3" : PATHS.inception_v3_weights,
    "DenseNet121" : PATHS.densenet121_weights,
    "InceptionResNetV2" : PATHS.inception_resnet_v2_weights,
}

pt_gradcam_layers = {
    "ResNet50": "conv5_block3_out",
    "InceptionV3": "mixed10",
    "DenseNet121": "conv5_block16_concat",
    "InceptionResNetV2": "conv_7b_ac", 
}