# src/model.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def build_unet(input_shape):
    """
    Builds a simple U-Net model.
    """
    inputs = Input(input_shape)

    # Encoder (Downsampling Path)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)

    # Decoder (Upsampling Path)
    u4 = UpSampling2D((2, 2))(c3)
    u4 = concatenate([u4, c2]) # Skip connection
    c4 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u4)
    
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c1]) # Skip connection
    c5 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)

    # Output Layer: 1 neuron with a sigmoid activation for binary segmentation
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

if __name__ == '__main__':
    # If this script is run directly, print the model summary.
    # This is a good way to test if the model architecture is defined correctly.
    from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    model = build_unet(input_shape)
    model.summary()
