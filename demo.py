import glob
import time

import cv2
import keras
import numpy as np


def predict(image, height=224, width=224):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    im = im / 255
    im = cv2.resize(im, (height, width))
    im = im.reshape((1,) + im.shape)
    
    pred = model.predict(im)
    
    mask = pred.reshape((224, 224))

    return mask


def transfer(image, mask):
    mask[mask > 0.5] = 255
    mask[mask <= 0.5] = 0

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    mask_n = np.zeros_like(image)
    mask_n[:, :, 0] = mask

    alpha = 0.8
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)

    return dst


if __name__ == '__main__':

    model = keras.models.load_model('models/hairnet_matting.hdf5')

    for name in glob.glob('test/images/*.jpg'):
        img = cv2.imread(name)

        st = time.time()

        mask = predict(img)

        d1 = time.time()
        dst = transfer(img, mask)

        print("segment: %f, color:%f" % (d1 - st, time.time() - d1))
        
        cv2.imwrite(name.replace('images', 'outs'), dst)
