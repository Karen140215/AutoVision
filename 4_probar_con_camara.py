"""Analiza la imagen de la cámara o de un video. Presione q o ESC para salir."""

from pathlib import Path
import time

import cv2
import numpy as np
import tensorflow as tf

# Entrada: True usa un video y False usa la cámara
USAR_VIDEO = True
NOMBRE_VIDEO = "Botella_con_sin_tapa_prueba_funcionamiento.mp4"
REPETIR_VIDEO = True

# Cámara: si no abre con 0, probar con 1
NUM_CAMARA = 0
ESPEJO = True
ANCHO_CAM, ALTO_CAM = 640, 480

# Tamaño de imagen y zona donde se mide el error
ANCHO, ALTO = 128, 128
FILA_INICIO, FILA_FIN = 16, 128
COL_INICIO, COL_FIN = 24, 104

# Carpetas y archivos
CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = CARPETA / "modelo_entrenado" / "autoencoder_botellas.keras"
RUTA_UMBRAL = CARPETA / "modelo_entrenado" / "umbral.txt"
RUTA_VIDEO = CARPETA / "videos" / "prueba" / NOMBRE_VIDEO


# Abre el video o la cámara
def abrir_entrada():
    if USAR_VIDEO:
        entrada = cv2.VideoCapture(str(RUTA_VIDEO))
        if not entrada.isOpened():
            raise SystemExit(f"No se pudo abrir el video: {RUTA_VIDEO}")
        print("Usando video:", NOMBRE_VIDEO)
        return entrada, "Camara simulada"

    entrada = cv2.VideoCapture(NUM_CAMARA)
    if not entrada.isOpened():
        raise SystemExit(f"No se pudo abrir la camara {NUM_CAMARA}")
    entrada.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAM)
    entrada.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAM)
    print("Usando camara", NUM_CAMARA)
    return entrada, "Camara"


# Lee un cuadro y reinicia el video cuando termina
def leer_cuadro(entrada):
    leido, cuadro = entrada.read()
    if not leido and USAR_VIDEO and REPETIR_VIDEO:
        entrada.set(cv2.CAP_PROP_POS_FRAMES, 0)
        leido, cuadro = entrada.read()
    if leido and ESPEJO and not USAR_VIDEO:
        cuadro = cv2.flip(cuadro, 1)
    return leido, cuadro


# Convierte el cuadro al formato que usa el modelo
def preparar_imagen(cuadro):
    imagen = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)
    imagen = cv2.resize(imagen, (ANCHO, ALTO))
    return np.expand_dims(imagen.astype("float32") / 255.0, axis=0)


# Calcula el error solo en la zona central
def calcular_mae(imagen, reconstruccion):
    diferencia = np.abs(imagen - reconstruccion)
    return float(np.mean(diferencia[:, FILA_INICIO:FILA_FIN, COL_INICIO:COL_FIN, :]))


def escribir(panel, texto, y, color, tamano=0.6):
    cv2.putText(panel, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, tamano, color, 2)


# Une la imagen, la reconstrucción y los resultados
def armar_pantalla(cuadro, reconstruccion, mae, umbral, fps, inferencia, titulo):
    entrada = cv2.resize(cuadro, (400, 400))
    salida = np.clip(reconstruccion[0] * 255, 0, 255).astype(np.uint8)
    salida = cv2.resize(cv2.cvtColor(salida, cv2.COLOR_RGB2BGR), (400, 400))

    escala = 400 / ANCHO
    esquina_1 = (int(COL_INICIO * escala), int(FILA_INICIO * escala))
    esquina_2 = (int(COL_FIN * escala) - 1, int(FILA_FIN * escala) - 1)
    cv2.rectangle(entrada, esquina_1, esquina_2, (0, 255, 255), 2)

    es_anomalia = mae > umbral
    estado = "ANOMALIA" if es_anomalia else "NO ANOMALIA"
    color_estado = (0, 0, 255) if es_anomalia else (0, 255, 0)

    escribir(entrada, titulo, 30, (255, 255, 255), 0.7)
    escribir(entrada, f"FPS: {fps:.1f}", 65, (0, 255, 255))
    escribir(salida, "Imagen reconstruida", 30, (255, 255, 255), 0.7)
    escribir(salida, f"Inferencia: {inferencia:.1f} ms", 65, (255, 255, 0))
    escribir(salida, f"MAE: {mae:.4f}  Umbral: {umbral:.4f}", 95, (255, 255, 0))
    escribir(salida, f"Estado: {estado}", 125, color_estado)

    return np.hstack((entrada, salida))


# Cargar el modelo y el umbral
print("Cargando modelo...")
modelo = tf.keras.models.load_model(RUTA_MODELO)
umbral = float(RUTA_UMBRAL.read_text())
print(f"Umbral: {umbral:.5f}")

# Abrir la entrada
entrada, titulo = abrir_entrada()
fps_video = entrada.get(cv2.CAP_PROP_FPS) or 20
print("Presione q o ESC para salir")

# Analizar cada cuadro
fps = 0
t_anterior = time.time()

while True:
    t_cuadro = time.time()
    leido, cuadro = leer_cuadro(entrada)
    if not leido:
        print("No se recibio imagen")
        break

    # Reconstruir la imagen
    imagen = preparar_imagen(cuadro)
    t_inicio = time.time()
    reconstruccion = modelo(imagen, training=False).numpy()
    inferencia = (time.time() - t_inicio) * 1000

    # Medir el error
    mae = calcular_mae(imagen, reconstruccion)

    t_actual = time.time()
    if t_actual > t_anterior:
        fps = 1 / (t_actual - t_anterior)
    t_anterior = t_actual

    # Mostrar el resultado
    pantalla = armar_pantalla(cuadro, reconstruccion, mae, umbral, fps, inferencia, titulo)
    cv2.imshow("Control de calidad - Autoencoder", pantalla)

    # Esperar una tecla: q o ESC para salir
    espera = 1
    if USAR_VIDEO:
        usado = (time.time() - t_cuadro) * 1000
        espera = max(1, int(1000 / fps_video - usado))

    tecla = cv2.waitKey(espera) & 0xFF
    if tecla in (ord("q"), 27):
        break

# Cerrar la entrada
entrada.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
