# BAGLS: Unhealthy vs Healthy Glottis Classification
An image classification project on the Benchmark of Automatic Glottis Segmentation (BAGLS) dataset.


## Full project report

`wandb.ai` Experiment report can be found in this link: [WANDB REPORT](https://wandb.ai/miked/bagls-sh-project/reports/Unhealthy-Glottis-Classification-BAGLS-dataset---VmlldzozMDgxNTkw?accessToken=rdn2slmxv3qjxomt8834r97yi0u00asw7d3aa3d2rnzw834whbawq12maihhz8ef).


## Model Details
- By: Michael Dorosan
- Date: Dec 4, 2022
- Model version: 0.0
- Model type: Classifer, `tensorflow.keras` `.h5` model
- Models made use of transfer learning which were initialized using pretrained weights of RadImagenet models.
    - Dataset details: [BAGLS paper](https://www.nature.com/articles/s41597-020-0526-3#Tab1)
    - Pretrained weights: [RadImagenet](https://github.com/BMEII-AI/RadImageNet)

### Intended Use
- Primary intended use: Experiment/Workflow Development [NOT INTENDED FOR PRODUCTION]


### Metrics

- FLOPS comparison

| Model                 | GFLOPs | Input Size  |
|-----------------------|--------|-------------|
| InceptionResNetV2     | 6.476  | [224,224,3] |
| DenseNet121           | 2.835  | [224,224,3] |
| InceptionV3           | 2.843  | [224,224,3] |
| ResNet50              | 3.863  | [224,224,3] |
| conv_net_from_scratch | 0.5581 | [224,224,3] |


### Evaluation Data
- Dataset: BAGLS `test` dataset
- Motivation: Determine model performance on a held out test set that preserves natural imbalance between classes
- Preprocessing. Only rescaling and resizing was done on the `test` subest provided by the BAGLS group. Images with unclear disorder status were removed.
    ```
    None (half year post-phonomicrosurgery for polypoid mid-membranous lesions)
    None (one year after presumption of a pseudocyst/sulcus in left vocal fold)
    functional
    <empty string>

    ```

### Training Data

- Dataset: BAGLS `train` dataset with less imbalanced class sizes (i.e., in terms of disorder status, image source, etc). 
- Motivation: Train model to learn from features of various classes. For this task, the classes were simplified to be either unhealthy or healthy glottis.
- Filtering: Images with unclear disorder status were removed.
    ```
    None (half year after diag of small vocal nodules)
    functional
    None (higher phonation)
    <empty string>

    ```
- Data Augmentation:
    - Image size: (224,224,3)
    - Rescale: 1 / 255 (normalization)
    - Rotation
    - Vertical flip
    - Translation
    - Shear
    - Zoom


### GradCAM Interpretation

GradCAM was used to interpret model predictions particularly those images with poor predictions of disorder status (false negative: predicted healthy but are otherwise not). Below is an example.

![sample prediction](sample-prediction.png)



## Setup
Create an environment for the project
```
conda env create -f environment.yml
```

Activate
```
conda activate bagls-sh-project
```

### Data Download

Generate an api token from your kaggle account and save the generated `kaggle.json` in the root directory of your machine `~/.kaggle/kaggle.json`.

Modify permissions to the file to secure the token and key `chmod 600 ~/.kaggle/kaggle.json`

The steps in setting up the kaggle public API are detailed in https://www.kaggle.com/docs/api.


Finally, download the dataset by running:

```
kaggle datasets download -d gomezp/benchmark-for-automatic-glottis-segmentation
```

The data will be downloaded to the current working directory as a `.zip` file. Unzip as shown:

```
unzip benchmark-for-automatic-glottis-segmentation.zip
```

Retain the resulting directory structure for use in the notebooks and scripts included in this repository.