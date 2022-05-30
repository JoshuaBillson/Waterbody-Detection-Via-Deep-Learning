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
    "alpha": 0.5,
    "learning_rate": 0.00005,
    "optimizer": "adam",
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

# Supported Bands

1. RGB
2. NIR
3. SWIR
4. RGB + NIR
5. RGB + SWIR *
6. NIR + SWIR *
7. RGB + NIR + SWIR *
