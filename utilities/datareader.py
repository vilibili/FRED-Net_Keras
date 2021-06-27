import os
import imageio
import numpy as np

image_path = r'dataset/images'
mask_path = r'dataset/masks'


def get_image_list(path):
    f = open(path, 'r')
    text = f.readlines()
    f.close()
    imagelist = []
    for name in text:
        name = name.replace('\n', '')
        if os.path.exists(os.path.join(image_path, name + '.jpeg')):
            imagelist.append(name)
        else:
            print('image is not found.', os.path.join(image_path, name + '.jpeg'))
        if os.path.exists(os.path.join(mask_path, name + '.png')) is False:
            print('mask is not found.', os.path.join(mask_path, name + '.png'))
    return imagelist

def get_Images(imagelistpath):
    images = []
    masks = []
    img_list = get_image_list(imagelistpath)
    num = len(img_list)
    print('number of images: ', num)
    for name in img_list:
        seg = np.zeros((256, 256, 2))
        image = imageio.imread(os.path.join(image_path, name + '.jpeg'))
        mask = imageio.imread(os.path.join(mask_path, name + '.png'))
        seg[:, :, 0] = mask == 0
        seg[:, :, 1] = mask == 255
        images.append(image)
        masks.append(seg)
    images = np.array(images)
    masks = np.array(masks)

    return images, masks