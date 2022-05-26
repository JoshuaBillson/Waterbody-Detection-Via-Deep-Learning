import json
from data_loader import DataLoader, create_patches, show_samples, load_dataset
from models import get_model
from models.losses import DiceBCELoss
from keras.metrics import MeanIoU, Recall, Precision
from keras.losses import BinaryCrossentropy
from models.layers import preprocessing_layer
from keras.layers import Input
from keras.models import Model
import numpy as np


def main():
    # Get Project Configuration
    with open('config.json') as f:
        config = json.loads(f.read())

    # Get Model Configuration
    with open('models.json') as f:
        model_config = json.loads(f.read())[config["hyperparameters"]["model"]]

    # Create Data Loader
    loader = DataLoader(timestamp=config["timestamp"])

    # Generate Patches
    if config["generate_patches"]:
        create_patches(loader, config["show_data"])

    # Show Samples Data
    if config["show_samples"]:
        show_samples(loader)

    # Load Dataset
    train_data, val_data, test_data = load_dataset(loader, batch_size=config["hyperparameters"]["batch_size"], include_nir=model_config["nir"], include_swir=model_config["swir"])
    #x, y = train_data[0]
    #print(x.shape, y.shape)
    #print(x.dtype, y.dtype)
    print(len(train_data))

    # Create Model
    '''
    inputs = Input(shape=(512, 512, 3))
    outputs = preprocessing_layer(inputs)
    #model = Model(inputs=inputs, outputs=outputs)
    model = get_model(config)
    model.summary()
    #print(train_data[0])
    x, y = train_data[0]
    print(x.shape, y.shape)
    p = model.predict(x)
    print(np.sum(y))
    print(DiceBCELoss(y, p))
    print(DiceBCELoss(y, y))
    #print(x, np.mean(x[0, :, :, 0]), np.std(x[0, :, :, 0]), np.mean(x[0, :, :, 1]), np.std(x[0, :, :, 1]), np.mean(x[0, :, :, 2]), np.std(x[0, :, :, 2]))
    #print(p, np.mean(p[0, :, :, 0]), np.std(p[0, :, :, 0]))
    #print(p[0, :, :, 0])
    #print(np.max(train_data[0]), np.max(p))
    '''
    model = get_model(config)
    model.summary()
    model.compile(loss=DiceBCELoss, optimizer="adam", metrics=[MeanIoU(num_classes=2), Precision(), Recall()])

    # Train Model
    if config["train"]:
        model.fit(train_data, epochs=config["hyperparameters"]["epochs"], verbose=1)


if __name__ == '__main__':
    main()
