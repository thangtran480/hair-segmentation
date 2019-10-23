import argparse

import keras

from data.load_data import trainGenerator
from nets import Hairnet

def get_args():
    parser = argparse.ArgumentParser(description='Hair Segmentation')
    parser.add_argument('--data_dir', default='./data/')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--use_pretrained', type=bool, default=False)
    parser.add_argument('--path_model', default='models/hair.h5')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()

    # Augmentation Data
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    
    myGene = trainGenerator(args.batch_size, args.data_dir, 'images', 'masks', data_gen_args, save_to_dir=None)

    model = keras.models.load_model(args.path_model)

    scores = model.evaluate_generator(myGene)

    print("scores:", scores)