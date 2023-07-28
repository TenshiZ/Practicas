from autodistill_yolov8 import YOLOv8
import os
import cv2
import numpy as np
import shutil

# Rutas de las carpetas
dataset_folder = "carpetas"
destination_folder = "datos_separados"

images_folder = os.path.join(destination_folder, "images")
labels_folder = os.path.join(destination_folder, "labels")
bounding_boxes_folder = os.path.join(destination_folder, "bounding_boxes")
cropped_images_folder = os.path.join(destination_folder, "cropped_images")

os.makedirs(images_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)
os.makedirs(bounding_boxes_folder, exist_ok=True)
os.makedirs(cropped_images_folder, exist_ok=True)

# Configuración de rutas y parámetros del modelo YOLOv3
target_model = YOLOv8("yolov8n.pt")
# Aquí debes definir el valor de DATA_YAML_PATH que no está definido en el código proporcionado
#target_model.train("dataset_clasi/data.yaml", epochs=200)


# Función para realizar la detección de objetos en una imagen
def detect_objects(image_path, class_name):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Obtener las detecciones del modelo YOLOv8
    detections = target_model.predict(image)

    class_labels = []
    bboxes = []
    for detection in detections:
        for obj in detection:
           for obj in detection:
            scores = obj[5:]
            if not scores:
                continue
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


def save_cropped_images(image_path, bboxes, class_name):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: No se pudo cargar la imagen '{image_path}'")
        return

    # Obtener la ruta de la carpeta específica de la especie dentro de "cropped_images"
    class_cropped_images_folder = os.path.join(cropped_images_folder, class_name)
    os.makedirs(class_cropped_images_folder, exist_ok=True)

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
            # Construir el nombre del archivo de imagen recortada
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            cropped_img_filename = f"{image_basename}_{i}.jpg"
            
            # Guardar la imagen recortada en la carpeta específica de la especie
            cropped_img_path = os.path.join(class_cropped_images_folder, cropped_img_filename)
            cv2.imwrite(cropped_img_path, cropped_img)


# Recorrer todas las imágenes en cada carpeta de especies
for root, _, files in os.walk(dataset_folder):
    for image_name in files:
        class_name = os.path.basename(root)
        image_path = os.path.join(root, image_name)
        class_labels, bboxes = detect_objects(image_path, class_name)

        # Obtener las dimensiones de la imagen
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Guardar imágenes en la carpeta "images"
        destination_image_path = os.path.join(images_folder, image_name)
        shutil.copy(image_path, destination_image_path)

        # Guardar etiquetas y cuadros delimitadores en formato YOLO
        # Aquí debes implementar la función save_labels_and_bboxes si es necesaria

        # Guardar imágenes recortadas en la carpeta "cropped_images"
        save_cropped_images(image_path, bboxes, class_name)
