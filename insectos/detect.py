import os
import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo preentrenado SSD MobileNetV2
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Función para realizar la detección de objetos en un directorio
def detect_objects_in_directory(directory_path, confidence_threshold=0.5):
    # Listar todos los subdirectorios que contienen las imágenes
    subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(directory_path, subdirectory)

        # Listar todas las imágenes en el subdirectorio
        image_files = [f for f in os.listdir(subdirectory_path) if f.endswith('.jpg') or f.endswith('.png')]

        for image_file in image_files:
            image_path = os.path.join(subdirectory_path, image_file)

            # Cargar la imagen
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Preprocesar la imagen para el modelo
            image = cv2.resize(image, (300, 300))
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            # Realizar la detección utilizando el modelo
            detections = model.predict(image)

            # Interpretar las detecciones
            num_detections = detections.shape[1]
            for i in range(num_detections):
                class_id = int(detections[0, i, 4])
                class_name = tf.keras.applications.imagenet_utils.decode_predictions(detections[:, :, 5:], top=1)[0][i][1]
                confidence = detections[0, i, 2]
                
                if confidence > confidence_threshold:
                    box = detections[0, i, :4] * 300
                    y_min, x_min, y_max, x_max = box.astype(np.int)
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(image, f'{class_name} - {confidence:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Mostrar la imagen con las detecciones
            cv2.imshow(f'Detected Objects - {image_file}', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Ejemplo de uso: Detectar objetos en todas las imágenes de los subdirectorios
directory_path = 'directorios'
detect_objects_in_directory(directory_path)
