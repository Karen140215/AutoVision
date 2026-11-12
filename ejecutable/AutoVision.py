"""AutoVision v1.0 - versión ejecutable de 4_probar_con_camara.py.

Al abrir, una ventana pregunta si se usa el video de prueba o la cámara
(o se indica al iniciar: AutoVision.exe --video  /  --camara 0).
En la ventana de resultados, presione q o ESC (o cierre la ventana) para salir.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np

# Video de prueba
NOMBRE_VIDEO = "Botella_con_sin_tapa_prueba_funcionamiento.mp4"
REPETIR_VIDEO = True

# Cámara
ESPEJO = True
ANCHO_CAM, ALTO_CAM = 640, 480

# Tamaño de imagen y zona donde se mide el error
ANCHO, ALTO = 128, 128
FILA_INICIO, FILA_FIN = 16, 128
COL_INICIO, COL_FIN = 24, 104

# Carpeta del programa: junto al .exe, o junto a este archivo si se corre con Python
if getattr(sys, "frozen", False):
    CARPETA = Path(sys.executable).resolve().parent
else:
    CARPETA = Path(__file__).resolve().parent

# Rutas relativas: OpenCV falla con rutas que tienen tildes o eñes,
# así que se trabaja desde la carpeta del programa
os.chdir(CARPETA)
RUTA_MODELO = Path("modelo_entrenado") / "autoencoder_botellas.keras"
RUTA_UMBRAL = Path("modelo_entrenado") / "umbral.txt"
RUTA_VIDEO = Path("videos") / "prueba" / NOMBRE_VIDEO

VENTANA = "AutoVision v1.0 - Control de calidad"


def error(mensaje):
    print("ERROR:", mensaje)
    raiz = tk.Tk()
    raiz.withdraw()
    messagebox.showerror("AutoVision", mensaje)
    raiz.destroy()
    sys.exit(1)


# Ventana inicial: elegir video o cámara
def elegir_entrada():
    eleccion = {}
    raiz = tk.Tk()
    raiz.title("AutoVision v1.0")
    raiz.resizable(False, False)

    tk.Label(raiz, text="AutoVision v1.0", font=("Segoe UI", 14, "bold")).pack(padx=30, pady=(18, 4))
    tk.Label(raiz, text="Detección de botellas sin tapa con autoencoder").pack(padx=30)
    tk.Label(raiz, text="¿Qué entrada desea usar?").pack(pady=(14, 6))

    def usar(tipo):
        eleccion["tipo"] = tipo
        eleccion["camara"] = int(num_camara.get())
        raiz.destroy()

    tk.Button(raiz, text="Video de prueba", width=24, command=lambda: usar("video")).pack(pady=3)

    fila = tk.Frame(raiz)
    fila.pack(pady=3)
    tk.Button(fila, text="Cámara", width=14, command=lambda: usar("camara")).pack(side="left")
    tk.Label(fila, text="  N.º").pack(side="left")
    num_camara = tk.Spinbox(fila, from_=0, to=9, width=3)
    num_camara.pack(side="left")

    tk.Label(raiz, text="Si la cámara no abre con 0, pruebe con 1.", fg="gray30").pack(pady=(8, 2))
    tk.Label(raiz, text="Para salir: q, ESC o cerrar la ventana.", fg="gray30").pack(pady=(0, 16))

    raiz.mainloop()
    return eleccion


# Abre el video o la cámara
def abrir_entrada(eleccion):
    if eleccion["tipo"] == "video":
        if not RUTA_VIDEO.exists():
            error(f"No se encontró el video:\n{CARPETA / RUTA_VIDEO}")
        entrada = cv2.VideoCapture(str(RUTA_VIDEO))
        if not entrada.isOpened():
            error(f"No se pudo abrir el video:\n{CARPETA / RUTA_VIDEO}")
        print("Usando video:", NOMBRE_VIDEO)
        return entrada, "Camara simulada"

    num = eleccion["camara"]
    entrada = cv2.VideoCapture(num)
    if not entrada.isOpened():
        error(f"No se pudo abrir la cámara {num}.\nPruebe con otro número.")
    entrada.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAM)
    entrada.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAM)
    print("Usando camara", num)
    return entrada, "Camara"


# Lee un cuadro y reinicia el video cuando termina
def leer_cuadro(entrada, usar_video):
    leido, cuadro = entrada.read()
    if not leido and usar_video and REPETIR_VIDEO:
        entrada.set(cv2.CAP_PROP_POS_FRAMES, 0)
        leido, cuadro = entrada.read()
    if leido and ESPEJO and not usar_video:
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


def leer_argumentos():
    # Opcional: AutoVision.exe --video   o   AutoVision.exe --camara 1
    args = sys.argv[1:]
    if "--video" in args:
        return {"tipo": "video", "camara": 0}
    if "--camara" in args:
        i = args.index("--camara")
        num = int(args[i + 1]) if i + 1 < len(args) and args[i + 1].isdigit() else 0
        return {"tipo": "camara", "camara": num}
    return None


def main():
    eleccion = leer_argumentos() or elegir_entrada()
    if not eleccion:
        return
    usar_video = eleccion["tipo"] == "video"

    # Cargar el modelo y el umbral
    if not RUTA_MODELO.exists() or not RUTA_UMBRAL.exists():
        error(f"No se encontró la carpeta modelo_entrenado junto al programa:\n{CARPETA}")
    print("Cargando modelo... (puede tardar unos segundos)")
    import tensorflow as tf
    modelo = tf.keras.models.load_model(RUTA_MODELO)
    umbral = float(RUTA_UMBRAL.read_text())
    print(f"Umbral: {umbral:.5f}")

    # Abrir la entrada
    entrada, titulo = abrir_entrada(eleccion)
    fps_video = entrada.get(cv2.CAP_PROP_FPS) or 20
    print("Presione q o ESC para salir")

    fps = 0
    t_anterior = time.time()

    while True:
        t_cuadro = time.time()
        leido, cuadro = leer_cuadro(entrada, usar_video)
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
        cv2.imshow(VENTANA, pantalla)

        # Esperar una tecla: q o ESC para salir
        espera = 1
        if usar_video:
            usado = (time.time() - t_cuadro) * 1000
            espera = max(1, int(1000 / fps_video - usado))

        tecla = cv2.waitKey(espera) & 0xFF
        if tecla in (ord("q"), 27):
            break
        # Salir también si se cierra la ventana con la X
        if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
            break

    # Cerrar la entrada
    entrada.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
