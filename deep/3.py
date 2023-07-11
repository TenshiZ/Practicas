import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings

def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed()

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')

warnings.filterwarnings('ignore')

(train_dataset, val_dataset), info = tfds.load('cifar10', split=['train', 'test'], with_info=True, as_supervised=True)

train_dataset = train_dataset.map(lambda x, y: (tf.image.resize(x, (224, 224)), y))
val_dataset = val_dataset.map(lambda x, y: (tf.image.resize(x, (224, 224)), y))

train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(32)

# Cargar el modelo pre-entrenado VGG16
pretrained_base = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar las capas del modelo pre-entrenado

pretrained_base.trainable = False

model = keras.Sequential([
    pretrained_base,
    layers.Conv2D(128, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_dataset, validation_data=val_dataset, epochs=2  )

# Obtener un lote de ejemplos del conjunto de datos de prueba
test_dataset = val_dataset.take(1)

# Realizar predicciones para el lote de imágenes
for images, labels in test_dataset:
    # Realizar las predicciones
    predictions = model.predict(images)

    # Mostrar cada imagen con su etiqueta verdadera y etiqueta predicha
    for i in range(images.shape[0]):
        image = images[i]
        true_label = info.features['label'].names[labels[i].numpy()]
        predicted_label = info.features['label'].names[np.argmax(predictions[i])]

        plt.imshow(image)
        plt.axis('off')
        plt.title(f'True: {true_label}, Predicted: {predicted_label}')
        plt.show()
        plt.savefig('imagen' + i)
