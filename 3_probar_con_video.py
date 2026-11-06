"""Analiza un video guardado. Presione q o ESC para salir."""

from pathlib import Path
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Configuración
NOMBRE_VIDEO = "Botella_con_sin_tapa_prueba_funcionamiento.mp4"
GUARDAR_VIDEO = True
MOSTRAR_VENTANA = "--sin-ventana" not in sys.argv

# Tamaño de imagen y zona donde se mide el error
ANCHO, ALTO = 128, 128
FILA_INICIO, FILA_FIN = 16, 128
COL_INICIO, COL_FIN = 24, 104

# Carpetas y archivos
CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = CARPETA / "modelo_entrenado" / "autoencoder_botellas.keras"
RUTA_UMBRAL = CARPETA / "modelo_entrenado" / "umbral.txt"
RUTA_VIDEO = CARPETA / "videos" / "prueba" / NOMBRE_VIDEO
CARPETA_RESULTADOS = CARPETA / "resultados"
RUTA_SALIDA = CARPETA_RESULTADOS / "resultado_prueba_video.mp4"


def verificar_archivos():
    for ruta in (RUTA_MODELO, RUTA_UMBRAL, RUTA_VIDEO):
        if not ruta.exists():
            raise SystemExit(f"No se encontró: {ruta}")


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
def armar_pantalla(cuadro, reconstruccion, mae, umbral, fps, inferencia):
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

    escribir(entrada, "Video de entrada", 30, (255, 255, 255), 0.7)
    escribir(entrada, f"FPS: {fps:.1f}", 65, (0, 255, 255))
    escribir(salida, "Imagen reconstruida", 30, (255, 255, 255), 0.7)
    escribir(salida, f"Inferencia: {inferencia:.1f} ms", 65, (255, 255, 0))
    escribir(salida, f"MAE: {mae:.4f}  Umbral: {umbral:.4f}", 95, (255, 255, 0))
    escribir(salida, f"Estado: {estado}", 125, color_estado)

    return np.hstack((entrada, salida))


# Muestra el resumen y guarda la gráfica del error
def graficar_mae(errores, umbral, fps_video):
    errores = np.array(errores)
    anomalias = errores > umbral
    tiempo = np.arange(len(errores)) / fps_video
    limite = errores.max() * 1.05

    print("\nResumen de la prueba:")
    print("Cuadros analizados:", len(errores))
    print("Cuadros sin anomalia:", int(np.sum(~anomalias)))
    print("Cuadros con anomalia:", int(np.sum(anomalias)))

    plt.figure(figsize=(12, 5))
    plt.plot(tiempo, errores, color="steelblue", linewidth=1.2, label="MAE por cuadro")
    plt.fill_between(tiempo, 0, limite, where=anomalias, color="red", alpha=0.12, label="Anomalia")
    plt.axhline(umbral, color="red", linestyle="--", label=f"Umbral: {umbral:.4f}")
    plt.xlabel("Tiempo del video (s)")
    plt.ylabel("MAE")
    plt.title("Error de reconstruccion por cuadro")
    plt.ylim(0, limite)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CARPETA_RESULTADOS / "5_mae_por_frame_video.png", dpi=120, bbox_inches="tight")
    plt.show()


# Cargar el modelo y el umbral
verificar_archivos()
CARPETA_RESULTADOS.mkdir(exist_ok=True)

print("Cargando modelo...")
modelo = tf.keras.models.load_model(RUTA_MODELO)
umbral = float(RUTA_UMBRAL.read_text())
print(f"Umbral: {umbral:.5f}")

# Abrir el video y preparar el archivo de salida
video = cv2.VideoCapture(str(RUTA_VIDEO))
if not video.isOpened():
    raise SystemExit(f"No se pudo abrir el video: {RUTA_VIDEO}")

fps_video = video.get(cv2.CAP_PROP_FPS) or 20
salida = None
if GUARDAR_VIDEO:
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    salida = cv2.VideoWriter(str(RUTA_SALIDA), codec, fps_video, (800, 400))

print("Procesando video...")
print("Presione q o ESC para salir")

# Analizar cada cuadro
errores = []
fps = 0
t_anterior = time.time()

while True:
    leido, cuadro = video.read()
    if not leido:
        print("Fin del video")
        break

    # Reconstruir la imagen
    imagen = preparar_imagen(cuadro)
    t_inicio = time.time()
    reconstruccion = modelo(imagen, training=False).numpy()
    inferencia = (time.time() - t_inicio) * 1000

    # Medir el error
    mae = calcular_mae(imagen, reconstruccion)
    errores.append(mae)

    t_actual = time.time()
    if t_actual > t_anterior:
        fps = 1 / (t_actual - t_anterior)
    t_anterior = t_actual

    # Mostrar y guardar el resultado
    pantalla = armar_pantalla(cuadro, reconstruccion, mae, umbral, fps, inferencia)
    if salida is not None:
        salida.write(pantalla)

    if MOSTRAR_VENTANA:
        cv2.imshow("Control de calidad - Autoencoder", pantalla)
        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (ord("q"), 27):
            break

# Cerrar los archivos
video.release()
if salida is not None:
    salida.release()
    print("Video guardado en:", RUTA_SALIDA)
cv2.destroyAllWindows()
cv2.waitKey(1)

# Mostrar el resumen final
graficar_mae(errores, umbral, fps_video)
