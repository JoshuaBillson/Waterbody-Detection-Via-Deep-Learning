# Waterbody Detection Via Deep Learning

This project explores the application of deep learning to waterbody detection.

# Dataset
https://drive.google.com/file/d/1faVYayxdNFGx2m0IxswDncoKmIxusdf7/view

# Running The Script
To run the script, simply execute `main.py` with the index of the GPU you want to train with specified as an optional parameter. If no GPU is specified, the script will default to GPU 0.
```bash
python3 main.py [GPU]
```

# Configuration
The script expects an external file called `config.json` in which the use should specify the desired configuration. Below is an example of such a file and a table outlining the effects of each setting.

### Example Configuration
```json
{
  "timestamp": 1,
  "patch_size": 512,
  "experiment_tag": "unet_multispectral",
  "create_logs": true,
  "train": true,
  "test": true,
  "experiments": 1,
  "use_mixed_precision": true,
  "hyperparameters": {
    "model": "unet", 
    "bands": ["RGB", "NIR", "SWIR"],
    "backbone": null,
    "learning_rate": 0.00005,
    "fusion_head": "naive",
    "loss": "jaccard_bce",
    "batch_size": 4,
    "epochs": 50,
    "apply_transfer": false,
    "random_subsample": false, 
    "water_threshold": 0
  }
}
```

### Available Settings
| Setting             | Effects                                                              |
|---------------------|----------------------------------------------------------------------|
| timestamp           | The timestamp to use (1, 2 or 3)                                     |
| patch_size          | The desired size of the generated patches                            |
| experiment_tag      | The human-readable tag with which to lable the experiment            |
| create_logs         | Indicates whether or not we want to create logs for the experiment   |
| train               | Whether or not we want to run the training loop                      |
| test                | Whether or not we want to test the trained model on the test set     |
| experiments         | Indicate the number of identical experiments we want to run          |
| use_mixed_precision | Indicate the number of identical experiments we want to run          |


### Available Hyperparameters
| Hyperparameter   | Effects                                                                           |
|------------------|-----------------------------------------------------------------------------------|
| model            | The model we want to use                                                          |
| bands            | The bands used as inpiut to the model                                             |
| backbone         | The model of the pre-trained backbone we want to use                              |
| learning_rate    | The learning rate used by the optimizer                                           |
| fusion_head      | The type of fusion head to use to combine spectral bands                          |
| loss             | The loss to use during training                                                   |
| batch_size       | The size of batches used in training                                              |
| epochs           | The number of epochs to train for                                                 |
| apply_transfer   | Whether or not to apply the PCT water transfer method                             |
| random_subsample | Whether or not to randomly sample patches for training                            |
| water_threshold  | The threshold at which to stop transplanting water bodies if apply_transfer=true  |

# Citation
Please cite our work if it is helpful for your research.
```
@article{rs15051253,
title={Water Body Extraction from Sentinel-2 Imagery with Deep Convolutional Networks and Pixelwise Category Transplantation},
author={Billson, Joshua and Islam, MD Samiul and Sun, Xinyao and Cheng, Irene},
journal={Remote Sensing},
volume={15},
year={2023},
number={5},
article-number={1253},
url={https://www.mdpi.com/2072-4292/15/5/1253},
issn={2072-4292},
doi={10.3390/rs15051253}
}
```
