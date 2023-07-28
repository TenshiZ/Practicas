from autodistill_yolov8 import YOLOv8


import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import shutil

BATCH_SIZE = 32
IMG_SIZE = (416, 416)

# Rutas de las carpetas
dataset_folder = "carpetas"
destination_folder = "datos_separados"

images_folder = os.path.join(destination_folder, "images")
labels_folder = os.path.join(destination_folder, "labels")
bounding_boxes_folder = os.path.join(destination_folder, "bounding_boxes")

os.makedirs(images_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)
os.makedirs(bounding_boxes_folder, exist_ok=True)

# Configuración de rutas y parámetros del modelo YOLOv3
target_model = YOLOv8("yolov8n.pt")
target_model.train(DATA_YAML_PATH, epochs=50)


# Función para realizar la detección de objetos en una imagen
def detect_objects(image_path, class_name):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]


    # Obtener las etiquetas y cuadros delimitadores
    class_labels = []
    bboxes = []
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:  # Prueba con un umbral más bajo (por ejemplo, 0.3)
                center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_labels.append(class_name)
                bboxes.append([x, y, x + w, y + h])
                print(f"Detected: {class_name} - Confianza: {confidence:.2f} - Cuadro delimitador: {x, y, x + w, y + h}")
            else:
                print(f"Ignorado: {class_name} - Confianza: {confidence:.2f}")

    return class_labels, bboxes


def save_cropped_images(image_path, bboxes):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: No se pudo cargar la imagen '{image_path}'")
        return

    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox

        # Asegúrate de que las coordenadas no estén fuera de los límites de la imagen
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)

        cropped_img = image[y_min:y_max, x_min:x_max]

        if cropped_img.size == 0:
            print(f"Advertencia: Cuadro delimitador inválido para la detección {i+1} en la imagen '{image_path}'")
        else:
            cropped_img_path = os.path.join(destination_folder, "cropped_images", f"{os.path.splitext(os.path.basename(image_path))[0]}_{i}.jpg")
            cv2.imwrite(cropped_img_path, cropped_img)



# Recorrer todas las imágenes en cada carpeta de especies
# Resto del código...

# Recorrer todas las imágenes en cada carpeta de especies
# Recorrer todas las imágenes en cada carpeta de especies
for class_name in os.listdir(dataset_folder):
    class_folder = os.path.join(dataset_folder, class_name)
    class_destination_folder = os.path.join(destination_folder, class_name)
   

    for image_name in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_name)
        class_labels, bboxes = detect_objects(image_path, class_name)

        # Obtener las dimensiones de la imagen
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Guardar imágenes en la carpeta "images"
        destination_image_path = os.path.join(images_folder, image_name)
        shutil.copy(image_path, destination_image_path)

        # Guardar etiquetas y cuadros delimitadores en formato YOLO
        #save_labels_and_bboxes(os.path.splitext(image_name)[0], class_labels, bboxes, width, height)

        # Guardar imágenes recortadas en la carpeta "cropped_images"
        save_cropped_images(image_name, bboxes)