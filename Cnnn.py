import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

#Contiene img de 5 tipos de flores creamos las etiquetas 
classes = ['roses', 'daisy', 'dandelion','sunflowers', 'tulips' ]


for cl in classes:
    img_path = os.path.join(base_dir, cl)
    images = glob.glob(img_path + '/*.jpg')
    print("{}: {} Images".format(cl, len(images)))
    num_train = int(round(len(images)*0.8))
    train, val = images[:num_train], images[num_train:]

for t in train:
    if not os.path.exists(os.path.join(base_dir, 'train', cl)):
        os.makedirs(os.path.join(base_dir, 'train', cl))
        shutil.move(t, os.path.join(base_dir, 'train', cl))

for v in val:
    if not os.path.exists(os.path.join(base_dir, 'val', cl)):
        os.makedirs(os.path.join(base_dir, 'val', cl))
        shutil.move(v, os.path.join(base_dir, 'val', cl))

    round(len(images)*0.8)

    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')


    batch_size = 100
    img_shape = 150 

    image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

train_data_gen = image_gen.flow_from_directory(
                                batch_size=batch_size,
                                directory=train_dir,
                                shuffle=True,
                                target_size=(img_shape,img_shape)
                                                )

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)


image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)

train_data_gen = image_gen.flow_from_directory(
                                batch_size=batch_size,
                                directory=train_dir,
                                shuffle=True,
                                target_size=(img_shape,img_shape)
                                                )

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)
    

image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)

train_data_gen = image_gen.flow_from_directory(
                                                batch_size=batch_size,
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=(img_shape, img_shape)
                                                )
"""use ImageDataGenerator para crear una transformación que cambie la escala de las imágenes en 255 y que se aplique:

rotación aleatoria de 45 grados
zoom aleatorio de hasta el 50%
volteo horizontal aleatorio
desplazamiento de ancho de 0,15
desplazamiento de altura de 0,15"""

image_gen = ImageDataGenerator(rescale=1./255, 
rotation_range=45, horizontal_flip=True, zoom_range=0.5, width_shift_range=15, height_shift_range=15  )


train_data_gen = image_gen.flow_from_directory(
                                                batch_size=batch_size,
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=(img_shape,img_shape),
                                                class_mode='sparse'
                                                )

#Validacion 
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=val_dir,
                                                 target_size=(img_shape, img_shape),
                                                 class_mode='sparse')                             

#CREAMOS LA CNN
model = Sequential()

model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_shape, img_shape,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(5))

#Compilar el modelo con Adam optimizer

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#Entrenar el modelo


epochs = 80

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / float(batch_size)))
)