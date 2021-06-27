import os
import numpy as np
import imageio
from model.SegNet import FRED_Net
import matplotlib.pyplot as plt

ckpt_path = r'ckpt/FRED-Net.h5'
image_path = r'dataset/images'
mask_path = r'dataset/masks'
image_name = 'sample'

model = FRED_Net()

if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)
    print('the checkpoint is loaded successfully.')

image = imageio.imread(os.path.join(image_path,image_name+'.jpeg'),as_gray=False, pilmode="RGB")
mask = imageio.imread(os.path.join(mask_path,image_name+'.png'))

result = model.predict(np.expand_dims(image,axis=0))
result = np.squeeze(result)
result = np.argmax(result, axis=2)

imageio.imsave(os.path.join('dataset',image_name+'.jpeg'),image,format='jpeg')
imageio.imsave(os.path.join('dataset',image_name+'.png'),mask,format='png')
imageio.imsave(os.path.join('dataset',image_name+'_predicted.png'),result,format='png')

plt.imshow(result)
plt.show()

plt.imshow(mask)
plt.show()