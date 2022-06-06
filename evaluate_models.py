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
from backend.config import get_epochs, get_model_type, get_timestamp, get_learning_rate
from callbacks import get_callbacks, create_callback_dirs

def test_preprocessing_layer():
    # Get Project Configuration
    with open('config.json') as f:
        config = json.loads(f.read())

    # Create Data Loader
    loader = DataLoader(timestamp=get_timestamp(config))

    # Load Dataset
    train_data, val_data, test_data = load_dataset(loader, config)
    print(len(train_data))

    inputs = Input(shape=(512, 512, 3))
    outputs = preprocessing_layer(inputs)
    model = Model(inputs=inputs, outputs=outputs)
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