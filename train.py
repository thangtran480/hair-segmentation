import argparse

from keras.callbacks import ModelCheckpoint
import keras
from keras.optimizers import Adam

from data.load_data import trainGenerator
from nets import Hairnet


def get_args():
    parser = argparse.ArgumentParser(description='Hair Segmentation')
    parser.add_argument('--data_dir', default='./data/')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--use_pretrained', type=bool, default=True)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    # Augmentation Data
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    
    myGene = trainGenerator(args.batch_size, args.data_dir, 'image', 'label', data_gen_args, save_to_dir=None)

    if args.use_pretrained:
        # Pretrain model
        model = keras.models.load_model('models/hairnet_matting.hdf5')
    else:
        model = Hairnet.get_model()

    model.compile(optimizer=Adam(lr=args.lr), loss='binary_crossentropy', metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint('models/hairnet_matting.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(myGene, callbacks=[model_checkpoint], steps_per_epoch=2000, epochs=args.epochs)

    model.save('models/hair.h5')
