# src/train.py

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Import configurations and functions from our other .py files
from config import TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, MODEL_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS
from data_loader import get_training_data
from model import build_unet

def main():
    """
    Main function to orchestrate the model training process.
    """
    print("--- Starting Model Training ---")

    # 1. Load and prepare the training and validation data
    X_train, X_val, y_train, y_val = get_training_data(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH)
    
    # 2. Define model input shape
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    
    # 3. Build the U-Net model
    model = build_unet(input_shape)
    
    # 4. Compile the model
    # We use BinaryCrossentropy for binary segmentation problems.
    # Adam is a standard, effective optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
                  
    print("Model has been compiled.")
    model.summary()

    # 5. Define callbacks for training
    # ModelCheckpoint saves the best model observed during training.
    checkpoint = ModelCheckpoint(MODEL_PATH, 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min')
    
    # EarlyStopping stops training if the validation loss doesn't improve for a certain number of epochs.
    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=5, 
                                   verbose=1, 
                                   restore_best_weights=True)

    callbacks_list = [checkpoint, early_stopping]
    
    # 6. Start training the model
    print("Starting training...")
    history = model.fit(X_train, y_train, 
                        batch_size=BATCH_SIZE, 
                        epochs=EPOCHS, 
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list)

    print("--- Training Finished ---")
    print(f"Best model saved to: {MODEL_PATH}")

if __name__ == '__main__':
    main()
