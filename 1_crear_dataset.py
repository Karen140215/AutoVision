"""Crea el dataset de fotos a partir de los videos de entrenamiento."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Configuración
VIDEOS = {
    "Botella_con_tapa": ["Botella_tapa1.mp4", "Botella_tapa2.mp4", "Botella_tapa3.mp4"],
    "Botella_sin_tapa": ["Botella_sin_tapa1.mp4", "Botella_sin_tapa2.mp4"],
}
FOTOS_POR_VIDEO = 200
TAMANO_FOTO = (320, 320)

# Carpetas del proyecto
CARPETA = Path(__file__).resolve().parent
CARPETA_VIDEOS = CARPETA / "videos" / "entrenamiento"
CARPETA_DATASET = CARPETA / "dataset_botellas"


# Guarda fotos repartidas a lo largo del video
def extraer_fotos(nombre_video, carpeta_salida):
    video = cv2.VideoCapture(str(CARPETA_VIDEOS / nombre_video))
    if not video.isOpened():
        print("No se pudo abrir:", nombre_video)
        return

    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    elegidos = set(np.linspace(0, total - 1, min(FOTOS_POR_VIDEO, total), dtype=int))

    numero = 0
    guardadas = 0
    while True:
        leido, imagen = video.read()
        if not leido:
            break

        if numero in elegidos:
            foto = cv2.resize(imagen, TAMANO_FOTO)
            nombre_foto = f"{Path(nombre_video).stem}_{numero:05d}.png"
            cv2.imwrite(str(carpeta_salida / nombre_foto), foto)
            guardadas += 1

        numero += 1

    video.release()
    print(f"{nombre_video}: {total} cuadros, {guardadas} fotos guardadas")


# Muestra 5 fotos de cada clase para revisarlas
def mostrar_muestras():
    figura, ejes = plt.subplots(2, 5, figsize=(14, 6))

    for fila, categoria in enumerate(VIDEOS):
        fotos = sorted((CARPETA_DATASET / categoria).glob("*.png"))
        if len(fotos) < 5:
            print("No hay suficientes fotos en", categoria)
            continue

        for columna, indice in enumerate(np.random.choice(len(fotos), 5, replace=False)):
            imagen = cv2.cvtColor(cv2.imread(str(fotos[indice])), cv2.COLOR_BGR2RGB)
            ejes[fila, columna].imshow(imagen)
            ejes[fila, columna].set_title(fotos[indice].stem, fontsize=8)
            ejes[fila, columna].axis("off")

    figura.suptitle("Muestras del dataset")
    plt.tight_layout()
    plt.show()


# Crear el dataset
for categoria, nombres in VIDEOS.items():
    carpeta_salida = CARPETA_DATASET / categoria
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    for nombre in nombres:
        extraer_fotos(nombre, carpeta_salida)

# Mostrar el resumen y las muestras
print("\nResumen del dataset:")
for categoria in VIDEOS:
    total_fotos = len(list((CARPETA_DATASET / categoria).glob("*.png")))
    print(f"{categoria}: {total_fotos} fotos")

mostrar_muestras()
