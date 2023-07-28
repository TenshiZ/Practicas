print("empezado++")
import tensorflow as tf
print("NP")
from autodistill import detection
print("NP1")
from autodistill_grounded_sam import GroundedSAM
print("NP2")
from IPython.display import Image
import cv2
import supervision as sv

#usar carpeta imagenes de division de datos ya que requiere todasjuntas

print("empezado")
ontology = detection.CaptionOntology({
    "insect": "insect",
    #promt, etiqueta que quieres
})

base_model = GroundedSAM(ontology=ontology) #crear
print("Creado")

test_image = 'carpetas/Diostrombus politus Uhler/004_5899.jpg'

image = cv2.imread(test_image) #imagen de prueba a elegir

classes = ["insect"] #ns si usar los proms o mis etiquetas, creo q proms

detections = base_model.predict(test_image)#predecir imagen a imagen , comprobar q va bien
print("Detecciones realizadas.")

box_annotator = sv.BoxAnnotator()

labels = [f"{classes[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]

annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

sv.plot_image(annotated_frame) #mostrar resultado


DATASET_DIR_PATH = "dataset_clasi" #donde guardar
IMAGE_DIR_PATH = "datos_separados/images"#de donde coger las imagenes
dataset = base_model.label( #predice la carpeta entera
    input_folder=IMAGE_DIR_PATH,  #path de carpeta de imagenes
    extension=".png", 
    output_folder=DATASET_DIR_PATH)#donde guardar

print("carpeta hecha")

#usando este dataset entrenarias a yolov8 que vayas a usar para detectar de verdad tus imagenes, aunq a nosotros nos serviria mas o menos con el anterior