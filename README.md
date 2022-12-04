# bagls-sh-project
An image classification project on the Benchmark of Automatic Glottis Segmentation (BAGLS) dataset.


## Setup
Create an environment for the project
```
conda env create -f environment.yml
```

Activate
```
conda activate cv-tf-torch
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