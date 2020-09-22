import glob
import time
import dlib

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import cv2
import keras
import numpy as np
import os


def getHead(hog_face_detector, image):
    faces_hog = hog_face_detector(image, 1)

    heads = []
    
    for face in faces_hog:
        
        head = dict()
        
        head["left"] = max(face.left() - 300, 0)
        head["top"] = max(face.top() - 300, 0)
        head["right"] = min(face.right() + 300, image.shape[0])
        head["bottom"] = min(face.bottom() + 300, image.shape[1])
        
        heads.append(head)

    return heads



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
#     os.environ["CUDA_VISIBLE_DEVICES"]= "1"
    
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#     config.log_device_placement = True  # to log device placement (on which device the operation ran)
#     sess = tf.Session(config=config)
#     set_session(sess)  # set this TensorFlow session as the default session for Keras        

    model = keras.models.load_model('/data.local/thangtv/hair-segmentation/checkpoints/06-06-2020_21-36-26/checkpoint.hdf5')
    hog_face_detector = dlib.get_frontal_face_detector()

    for path_image in glob.glob('test/images/*'):
        image = cv2.imread(path_image)
        heads = getHead(hog_face_detector, image)

        for head in heads:
            img = image[head["top"]:head["bottom"], head["left"]:head["right"]]
        
            mask = predict(img)

            dst = transfer(img, mask)

            cv2.imwrite(path_image.replace('images', 'outs'), dst)
