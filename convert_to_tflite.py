import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file("models/hairnet_matting_30.hdf5")
tflite_model = converter.convert()
open("models/converted_model_hairnet.tflite", "wb").write(tflite_model)