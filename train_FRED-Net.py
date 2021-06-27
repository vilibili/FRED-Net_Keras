import os
import keras
import imageio
import numpy as np
from keras.optimizers import *
import matplotlib.pyplot as plt
from utilities import datareader
from model.SegNet import FRED_Net
from keras.callbacks import ModelCheckpoint,EarlyStopping

class loss_history(keras.callbacks.Callback):
    def __init__(self, x=4):
        self.x = x
    def on_epoch_begin(self, epoch, logs={}):
        test_image_path = r'dataset\images\S4090L04_03311.jpeg'
        test_image = imageio.imread(test_image_path,as_gray=False, pilmode="RGB")

        pred = self.model.predict([np.expand_dims(test_image, axis=0)],verbose=1)
        pred = np.squeeze(pred)
        pred = np.argmax(pred, axis=2)
        plt.imshow(pred)
        plt.show()

ckpt_path = r'ckpt/FRED-Net.h5'
image_list = r'dataset/train.txt'
model = FRED_Net()
model.summary()
model.compile(optimizer=SGD(lr=0.1, decay=0.0005), loss=['categorical_crossentropy'], metrics=['accuracy'])
earlystopper = EarlyStopping(patience=7, verbose=1)
checkpointer = ModelCheckpoint(ckpt_path,verbose=1,save_best_only=True)

if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)
    print('the checkpoint is loaded successfully.')

images, masks = datareader.get_Images(image_list)
val_images, val_masks = datareader.get_Images(r'dataset/val.txt')
model.fit(x=images,
          y=masks,
          validation_data=(val_images,val_masks),
          callbacks=[loss_history(), checkpointer, earlystopper],
          epochs=100,
          verbose=1,
          batch_size=2,
          shuffle=True)
print('the checkpoint is saved successfully. Path : ' + ckpt_path)