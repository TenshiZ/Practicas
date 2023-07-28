
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import math, re, os,random,shutil, csv
import matplotlib.pyplot as plt
import warnings
import glob
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import pandas as pd
from collections import defaultdict

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import Xception

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


dataset = []
image_paths = []
labels = []



BATCH_SIZE = 128
IMG_SIZE = (224, 224)

# Create the combined dataset from the directory
dataset = tf.keras.utils.image_dataset_from_directory(
    'datos_separados/cropped_images',
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)   


class_names = dataset.class_names #nombres de las clases

train_dataset = tf.keras.utils.image_dataset_from_directory(
    'directoriosS/train',
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    'directoriosS/val',
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    'directoriosS/test',
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
])


initial_epochs = 1000

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience = 100,
    min_delta = 0.001,
    restore_best_weights = True,
)
nombre_guardado_congelado = 'species_recognition_model_xception_le_1'
checkpoint_callback = ModelCheckpoint(
   filepath = nombre_guardado_congelado + '.h5',
   save_weights_only = False,
   sabe_best_only = True,
   save_freq = 'epoch'
)

preentreno = "xception"
preprocess_input =Xception(weights='imagenet', include_top=False)

IMG_SHAPE = IMG_SIZE + (3,)
pretrained_base = tf.keras.applications.Xception(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


pretrained_base.trainable = False
mutaciones = "0.1_l2, learning_1"


model = keras.Sequential([
    pretrained_base,
    data_augmentation,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.1)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])



def exponential_lr(epoch,
                   start_lr = 0.00001, min_lr = 0.00001, max_lr = 0.00005,
                   rampup_epochs = 5, sustain_epochs = 0,
                   exp_decay = 0.8):

    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        # linear increase from start to rampup_epochs
        if epoch < rampup_epochs:
            lr = ((max_lr - start_lr) /
                  rampup_epochs * epoch + start_lr)
        # constant max_lr during sustain_epochs
        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr
        # exponential decay towards min_lr
        else:
            lr = ((max_lr - min_lr) *
                  exp_decay**(epoch - rampup_epochs - sustain_epochs) +
                  min_lr)
        return lr
    return lr(epoch,
              start_lr,
              min_lr,
              max_lr,
              rampup_epochs,
              sustain_epochs,
              exp_decay)

lr_callback = tf.keras.callbacks.LearningRateScheduler(exponential_lr, verbose=True)

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset,
                    callbacks=[early_stopping,checkpoint_callback,lr_callback],)



model.save('modelos/' + nombre_guardado_congelado + '.h5')
pretrained_base.trainable = True

print("Number of layers in the base model: ", len(pretrained_base.layers))
part_congelada = len(pretrained_base.layers)*0.10
print("Number of layers in the base model  congelados: ", (int(part_congelada)))
for layer in pretrained_base.layers[:int(part_congelada)]:
  layer.trainable = False

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate/10),
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)

epochs_cont = initial_epochs + 1000


nombre_guardado = nombre_guardado_congelado + '_2'

checkpoint_callback = ModelCheckpoint(
   filepath = nombre_guardado + '.h5',
   save_weights_only = False,
   sabe_best_only = True,
   save_freq = 'epoch'
)

history = model.fit(train_dataset,
                    initial_epoch=history.epoch[-1],
                    epochs=epochs_cont,
                    validation_data=validation_dataset,
                    callbacks=[early_stopping, checkpoint_callback],)


model.save('modelos/' + nombre_guardado + '.h5')

true_labels = []
predicted_labels = []

for image, label in test_dataset:
    true_labels.append(label)
    predicted_probs = model.predict(image)  # Realiza la predicción utilizando tu modelo
    predicted_label = np.argmax(predicted_probs, axis=1)
    predicted_labels.append(predicted_label)

true_labels = np.concatenate(true_labels)
predicted_labels = np.concatenate(predicted_labels)
#cmat = confusion_matrix(true_labels, predicted_labels, normalize='all')
cmat = confusion_matrix(true_labels, predicted_labels)


loss, accuracy ,= model.evaluate(test_dataset)
score = f1_score(true_labels, predicted_labels, labels=labels, average='macro')
precision = precision_score(true_labels, predicted_labels, labels=labels, average='macro')
recall = recall_score(true_labels, predicted_labels, labels=labels, average='macro')

results = [

    [preentreno, accuracy, precision, recall, score, nombre_guardado, mutaciones]
]

filename = 'results.csv'

with open(filename, mode='a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(results)

print('Los resultados se han guardado en', filename)