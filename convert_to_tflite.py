import tensorflow as tf

if __name__ == '__main__':
    # https://www.tensorflow.org/lite/performance/post_training_quantization#float16_quantization_of_weights

    tf.enable_eager_execution()

    PATH_MODEL = "models/hair.h5"
    PATH_MODEL_TFLITE = "models/model_hairnet.tflite"

    converter = tf.lite.TFLiteConverter.from_keras_model_file(PATH_MODEL)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]

    tflite_model = converter.convert()
    open(PATH_MODEL_TFLITE, "wb").write(tflite_model)
