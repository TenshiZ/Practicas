import cv2
#usar yolo_segmentation import YOLOSegmentation, descargar
img = cv2.imread("carpetas/Aulacophora indica (Gmelin)/002_1086.jpg")
img = cv2.resize(img, None, fx = 0.7, fy = 0.7)

#segmentation detector

ys = YOLOSegmentation("yolov8m-seg.pt")

bboxes, classes, segmentations, scores = ys.detect(img)

for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
    # print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
    (x, y, x2, y2) = bbox
    if class_id == 32: # q es lo q quiere detectar
        cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)
        cv2.polylines(img, [seg], True, (0, 0, 255), 4)
        cv2.putText(img, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    


cv2.imshow("carpetas/Aulacophora indica (Gmelin)/002_1086.jpg", img)
cv2.waitKey(0)

import cv2
import numpy as np

# Cargar la imagen y las máscaras (supongamos que tienes 'image' y 'masks')
# image: Imagen original (numpy array)
# masks: Lista de máscaras generadas por el modelo (numpy arrays)

# Crear una máscara final combinando todas las máscaras
final_mask = np.zeros_like(masks[0])
for mask in masks:
    final_mask = np.logical_or(final_mask, mask)

# Aplicar la máscara a la imagen original
segmented_image = cv2.bitwise_and(image, image, mask=final_mask.astype(np.uint8))

# Encontrar el contorno de la región segmentada
contours, _ = cv2.findContours(final_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Recortar la región segmentada de la imagen original
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cropped_segment = segmented_image[y:y+h, x:x+w]

    # Mostrar el resultado
    cv2.imshow('Región Segmentada', cropped_segment)
    cv2.waitKey(0)

cv2.destroyAllWindows()
