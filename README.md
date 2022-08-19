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
| patch_size       | The desired size of the generated patches        | {128, 256, 512} |
| train            | Whether or not we want to run the training loop  | Boolean         |

### Available Hyperparameters
| Hyperparameter   | Effects                                               |  Values                                         |
|------------------|-------------------------------------------------------|:-----------------------------------------------:|
| model            | The model we want to use                              | String                                          |
| bands            | The bands used as inpiut to the model                 | List<String>                                    |
| backbone         | The model of the pre-trained backbone we want to use  | String                                          |
| learning_rate    | The learning rate used by the optimizer               | Non-Zero Positive Float                         |
| batch_size       | The size of batches used in training                  | Non-Zero Positive Integer                       |
| epochs           | The number of epochs to train for                     | Non-Zero Positive Integer                       |

### Additional Notes
`model` Can be either the name of a base model in the set `unet`, `vnet`, `unet_plus`, `unet_3_plus`, `r2_unet`, `resunet`, `u2net`, `transunet`, `swin_unet`, `att_unet`, `fpn`, `link_net`, `psp_net` or an existing checkpointed model such as `unet.nir.none.1653771136`. In the case of the former, a new named model will be constructed and checkpointed. In the case of the latter, the saved model will be reinitialized from its saved weights.  

`bands` Is a non-empty list containing any combination of the strings in the set {"RGB", "NIR", "SWIR"}.  

`backbone` A string belonging to the set `VGG[16, 19]`, `ResNet[50,101,152]`, `ResNet[50,101,152]V2`, `DenseNet[121,169,201]`, `EfficientNetB[0-7]` or `null` if you don't want to use a pre-trained backbone.  


# Model Naming Conventions
All saved models are named following the convention `{base_model}.{bands}.{backbone}.{time}`. So a model based on U-Net++ with a trained ResNet152 backbone taking the RGB and NIR bands as input would be named `unet_plus.rgb+nir.resnet152.1653771136`.  

# Logs
Training performance is logged every epoch with both TensorBoard and CSV. If loading a previously trained model, the logs for the new epochs will be appended to the existing logs.  
  
