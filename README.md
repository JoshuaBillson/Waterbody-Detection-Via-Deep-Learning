# Waterbody Detection Via Deep Learning

This project explores the application of deep learning to waterbody detection.

# Dataset
https://github.com/SCoulY/Sentinel-2-Water-Segmentation

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
  "generate_patches": false,
  "show_data": false,
  "show_samples": false,
  "timestamp": 1,
  "patch_size": 512,
  "train": true,
  "hyperparameters": {
    "model": "unet",
    "bands": ["NIR"],
    "backbone": null,
    "learning_rate": 0.00005,
    "batch_size": 4,
    "epochs": 5 }
}
```

### Available Settings
| Setting          | Effects                                          |  Values         |
|------------------|--------------------------------------------------|:---------------:|
| generate_patches | Generate patches and save to disk                | Boolean         |
| show_data        | Visualize the initial dataset                    | Boolean         |
| show_samples     | Visualize a sample of patches                    | Boolean         |
| timestamp        | The timestamp we want to use for the dataset     | {1, 2, 3}       |
| patch_size       | The desired of the generated patches             | {128, 256, 512} |
| train            | Whether or not we want to run the training loop  | Boolean         |

### Available Hyperparameters
| Hyperparameter   | Effects                                               |  Values                                         |
|------------------|-------------------------------------------------------|:-----------------------------------------------:|
| model            | The model we want to use                              | Name of either a base or checkpointed model     |
| bands            | The bands used as inpiut to the model                 | A list containing any of {"RGB", "NIR", "SWIR"} |
| backbone         | The model of the pre-trained backbone we want to use  | Name of the backbone ("ResNet152", etc.)        |
| learning_rate    | The learning rate used by the optimizer               | Non-Zero Positive Float                         |
| batch_size       | The size of batches used in training                  | Non-Zero Positive Integer                       |
| epochs           | The number of epochs to train for                     | Non-Zero Positive Integer                       |

# Supported Bands

1. RGB
2. NIR
3. SWIR
4. RGB + NIR
5. RGB + SWIR *
6. NIR + SWIR *
7. RGB + NIR + SWIR *
