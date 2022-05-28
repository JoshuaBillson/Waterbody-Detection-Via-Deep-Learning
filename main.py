import os
import json
from tensorflow.keras.metrics import MeanIoU, Recall, Precision
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from models.losses import DiceBCELoss
from data_loader import DataLoader, create_patches, show_samples, load_dataset
from models.layers import preprocessing_layer
from models import get_model
from config import get_epochs, get_model_type, get_timestamp, get_learning_rate
from callbacks import get_callbacks, create_callback_dirs


GPU = 5

def main():
    # Get Project Configuration
    with open('config.json') as f:
        config = json.loads(f.read())

    # Create Data Loader
    loader = DataLoader(timestamp=get_timestamp(config))

    # Generate Patches
    if config["generate_patches"]:
        create_patches(loader, config["show_data"])

    # Show Samples Data
    if config["show_samples"]:
        show_samples(loader)

    # Load Dataset
    train_data, val_data, test_data = load_dataset(loader, config)
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
    model.compile(loss=DiceBCELoss, optimizer=Adam(learning_rate=get_learning_rate(config)), metrics=[MeanIoU(num_classes=2), Precision(), Recall()])

    # Get Callbacks
    create_callback_dirs(config)
    callbacks = get_callbacks(config)

    # Train Model
    if config["train"]:
        model.fit(train_data, epochs=get_epochs(config), verbose=1, callbacks=callbacks, validation_data=val_data)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{GPU}"
    main()
