from keras.callbacks import ModelCheckpoint
import keras 
from data.load_data import trainGenerator
from nets import Hairnet

if __name__ == '__main__':
    BATCH_SIZE = 4
    DATA_PATH = 'data/'

    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    myGene = trainGenerator(BATCH_SIZE, DATA_PATH, 'image', 'label', data_gen_args,
                            save_to_dir=None)

	model = Hairnet.get_model()
	
	# Pretrain model 
    # model = keras.models.load_model('hairnet_matting3.hdf5')

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint('hairnet_matting.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(myGene, callbacks=[model_checkpoint], steps_per_epoch=2000, epochs=30)
