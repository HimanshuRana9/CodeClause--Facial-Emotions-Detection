# train.py - minimal training script (expects image folders or FER CSV)
import os
from model import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def main():
    # This script is a template. To use it:
    # - Prepare image folders: data/train/<label> and data/val/<label>
    # - Or adapt to load FER2013 CSV.
    train_dir = os.path.join('data', 'train')
    val_dir = os.path.join('data', 'val')
    if not os.path.exists(train_dir):
        print('No training data found at data/train. Please prepare dataset or use the included sample model for demo.')
        return
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(train_dir, target_size=(48,48), color_mode='grayscale', class_mode='categorical', batch_size=32)
    val_gen = val_datagen.flow_from_directory(val_dir, target_size=(48,48), color_mode='grayscale', class_mode='categorical', batch_size=32)
    model = build_model()
    callbacks = [tf.keras.callbacks.ModelCheckpoint('emotion_model.h5', save_best_only=True, monitor='val_accuracy')]
    model.fit(train_gen, validation_data=val_gen, epochs=15, callbacks=callbacks)
    print('Training finished. Best model saved to emotion_model.h5')

if __name__ == '__main__':
    main()
