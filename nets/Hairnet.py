from keras import layers, Model
from keras.optimizers import Adam

image_data_format = 'channels_last'


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if image_data_format == 'channels_first' else -1

    x = layers.ZeroPadding2D(padding=(1, 1), data_format=image_data_format, name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      data_format=image_data_format,
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  momentum=0.1,
                                  epsilon=1e-05,
                                  name='conv1_bn')(x)
    x = layers.ReLU(6., name='conv1_relu')(x)

    return x


def _depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = 1 if image_data_format == 'channels_first' else -1

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(padding=(1, 1),
                                 data_format=image_data_format,
                                 name='conv_pad_%d' % block_id)(inputs)

    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               data_format=image_data_format,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-05,
                                  momentum=0.1,
                                  name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      data_format=image_data_format,
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-05,
                                  momentum=0.1,
                                  name='conv_pw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

    return x


def YellowBlock(inputs):
    return layers.UpSampling2D(size=(2, 2), data_format=image_data_format)(inputs)


def OrangeBlock(inputs, filters, block_id, kernel_size=(3, 3), stride=(1, 1)):
    x = layers.ZeroPadding2D(padding=(1, 1), data_format=image_data_format, name='conv_pad_orange_%d' % block_id)(
        inputs)
    x = layers.SeparableConv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=stride,
                               use_bias=False,
                               data_format=image_data_format,
                               name='sep_conv_orange_%d' % block_id)(x)

    x = layers.ReLU()(x)

    return x


def RedBlock(inputs, filters, kernel_size=(1, 1)):
    x = layers.Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=(1, 1),
                    #   activation='softmax',
                      data_format=image_data_format)(inputs)

    return x


def get_model(input_shape=(224, 224, 3)):
    img_input = layers.Input(shape=input_shape)

    x = _conv_block(img_input, 32, strides=(2, 2))
    skip1 = _depthwise_conv_block(x, 64, block_id=1)  # skip1 1

    x = _depthwise_conv_block(skip1, 128, strides=(2, 2), block_id=2)
    skip2 = _depthwise_conv_block(x, 128, block_id=3)  # skip2 3

    x = _depthwise_conv_block(skip2, 256, strides=(2, 2), block_id=4)
    skip3 = _depthwise_conv_block(x, 256, block_id=5)  # skip3 5

    x = _depthwise_conv_block(skip3, 512, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, block_id=7)
    x = _depthwise_conv_block(x, 512, block_id=8)
    x = _depthwise_conv_block(x, 512, block_id=9)
    x = _depthwise_conv_block(x, 512, block_id=10)
    skip4 = _depthwise_conv_block(x, 512, block_id=11)  # skip4 11

    x = _depthwise_conv_block(skip4, 1024, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, block_id=13)
    x = _depthwise_conv_block(x, 1024, block_id=14)

    x = YellowBlock(x)
    x = layers.Concatenate(axis=-1)([x, skip4])
    x = OrangeBlock(x, 64, 1)

    x = YellowBlock(x)
    x = layers.Concatenate(axis=-1)([x, skip3])
    x = OrangeBlock(x, 64, 2)

    x = YellowBlock(x)
    x = layers.Concatenate(axis=-1)([x, skip2])
    x = OrangeBlock(x, 64, 3)

    x = YellowBlock(x)
    x = layers.Concatenate(axis=-1)([x, skip1])
    x = OrangeBlock(x, 64, 4)

    x = YellowBlock(x)
    x = layers.Concatenate(axis=-1)([x, img_input])
    x = OrangeBlock(x, 64, 5)

    x = RedBlock(x, 1)

    model = Model(img_input, x)

    return model

if __name__ == '__main__':

    model = get_model()
    print(model.summary())