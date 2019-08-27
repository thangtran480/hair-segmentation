import tensorflow as tf


if __name__ == '__main__':

    PATH_MODEL_TFLITE = "models/model_hairnet.tflite"

    interpreter = tf.lite.Interpreter(model_path=PATH_MODEL_TFLITE)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input Shape: {}, Output Shape: {}".format(input_details[0]['shape'], output_details[0]['shape']))
