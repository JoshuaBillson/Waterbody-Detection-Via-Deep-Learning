from tensorflow import keras
from models.layers import fusion_head, preprocessing_layer
from keras.layers import Conv2D, Concatenate, Layer, Input, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from keras.models import Model
from keras_unet_collection.models import unet_2d


def unet_multispectral(config) -> Model:
    # Change Model Output
    model = unet_2d(input_size=(config["patch_size"] // 2, config["patch_size"] // 2, 128), filter_num=[64, 128, 256, 512, 1024], n_labels=2, backbone=None)
    x = model.layers[-3].output  # fetch the last layer previous layer output
    up_sample = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same')(x)
    outputs = Conv2D(1, kernel_size=(1, 1), name="out", activation='sigmoid', dtype="float32")(up_sample)  # create new last layer
    model = Model(inputs=model.input, outputs=outputs)
    model.summary()

    # Change Model Input
    fusion_input, fusion_output = fusion_head(patch_size=config["patch_size"])
    outputs = model(fusion_output)
    return Model(inputs=fusion_input, outputs=outputs)


def unet_rgb(config) -> Model:
    # Change Model Output
    model = unet_2d(input_size=(config["patch_size"], config["patch_size"], 3), filter_num=[64, 128, 256, 512, 1024], n_labels=2, backbone="ResNet152")
    x = model.layers[-3].output  # fetch the last layer previous layer output
    outputs = Conv2D(1, kernel_size=(1, 1), name="out", activation='sigmoid', dtype="float32")(x)  # create new last layer
    model = Model(inputs=model.input, outputs=outputs)

    # Change Model Input
    inputs, out = preprocessing_layer(config["patch_size"])
    outputs = model(out)
    return Model(inputs=inputs, outputs=outputs)


def unet(inputs: Layer) -> Model:
    """
        Summary:
            Create UNET model object
        Arguments:
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    # Contraction path
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path
    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)  # Original 0.1
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)  # Original 0.1
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    return Conv2D(1, (1, 1), activation='sigmoid', dtype='float32')(c9)
