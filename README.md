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
| Hyperparameter   | Effects                                               |  Values                                         |
|------------------|-------------------------------------------------------|:-----------------------------------------------:|
| model            | The model we want to use                              | String                                          |
| bands            | The bands used as inpiut to the model                 | List<String>                                    |
| backbone         | The model of the pre-trained backbone we want to use  | String                                          |
| learning_rate    | The learning rate used by the optimizer               | Non-Zero Positive Float                         |
| fusion_head      | The learning rate used by the optimizer               | Non-Zero Positive Float                         |
| loss             | The learning rate used by the optimizer               | Non-Zero Positive Float                         |
| batch_size       | The size of batches used in training                  | Non-Zero Positive Integer                       |
| epochs           | The number of epochs to train for                     | Non-Zero Positive Integer                       |
| apply_transfer   | The number of epochs to train for                     | Non-Zero Positive Integer                       |
| random_subsample | The number of epochs to train for                     | Non-Zero Positive Integer                       |
| water_threshold  | The number of epochs to train for                     | Non-Zero Positive Integer                       |
  
