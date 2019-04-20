import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop,Adam,SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

img_width = 20
img_height = 20
train_data_dir = "data"
validation_data_dir = "test"
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.1,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)
val_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.1,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = 5,
class_mode = "categorical")
validation_generator = val_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")


net = Sequential()
net.add(Conv2D(32,kernel_size = 5, input_shape=(img_height,img_width,3), padding='same', activation='relu'))
net.add(MaxPooling2D(pool_size=(2, 2)))
net.add(Conv2D(64, kernel_size = 5, activation='relu', padding='same'))
net.add(MaxPooling2D(pool_size=(2, 2)))
net.add(Flatten())
net.add(Dense(256, activation='relu'))
net.add(Dense(128, activation='relu'))
#net.add(Dropout(0.5))
net.add(Dense(34, activation='softmax'))

net.summary()

checkpoint = ModelCheckpoint("model.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
net.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
net.fit_generator(train_generator, samples_per_epoch = 1639, epochs = 50, validation_data = validation_generator,
                        nb_val_samples = 447, callbacks = [checkpoint, early])

from keras.models import model_from_json
model_json = net.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
net.save_weights("model.h5")
print("Saved model to disk")
