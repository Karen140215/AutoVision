"""Entrena el autoencoder con botellas con tapa y calcula el umbral."""

from pathlib import Path
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Configuración
ALTO, ANCHO, CANALES = 128, 128, 3
TAMANO_CODIGO = 64
EPOCAS = 150
LOTE = 32
NUM_DESVIACIONES = 3
FILA_INICIO, FILA_FIN = 16, 128
COL_INICIO, COL_FIN = 24, 104

# Carpetas y archivos
CARPETA = Path(__file__).resolve().parent
CARPETA_DATASET = CARPETA / "dataset_botellas"
CARPETA_RESULTADOS = CARPETA / "resultados"
CARPETA_MODELO = CARPETA / "modelo_entrenado"
RUTA_MODELO = CARPETA_MODELO / "autoencoder_botellas.keras"
RUTA_UMBRAL = CARPETA_MODELO / "umbral.txt"

CARPETA_RESULTADOS.mkdir(exist_ok=True)
CARPETA_MODELO.mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(42)


# Carga las fotos a color, en 128 x 128 y con valores entre 0 y 1
def cargar_fotos(carpeta):
    fotos = []
    for ruta in sorted(carpeta.glob("*.png")):
        imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
        if imagen is None:
            continue
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        imagen = cv2.resize(imagen, (ANCHO, ALTO))
        fotos.append(imagen.astype("float32") / 255.0)
    return np.array(fotos, dtype="float32")


# Calcula el error solo en la zona central
def calcular_mae(originales, reconstruidas):
    diferencia = np.abs(originales - reconstruidas)
    zona = diferencia[:, FILA_INICIO:FILA_FIN, COL_INICIO:COL_FIN, :]
    return np.mean(zona, axis=(1, 2, 3))


# Crea el modelo y lo prepara para entrenar
def crear_autoencoder():
    modelo = models.Sequential([
        layers.Input(shape=(ALTO, ANCHO, CANALES)),

        # Codificador
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Cuello de botella de 64 valores
        layers.Flatten(),
        layers.Dense(TAMANO_CODIGO, activation="relu"),
        layers.Dense(8 * 8 * 128, activation="relu"),
        layers.Reshape((8, 8, 128)),

        # Decodificador
        layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation="relu", padding="same"),
        layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation="relu", padding="same"),
        layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation="relu", padding="same"),
        layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation="relu", padding="same"),
        layers.Conv2D(CANALES, (3, 3), activation="sigmoid", padding="same"),
    ])
    modelo.compile(optimizer="adam", loss="mae")
    return modelo


# Guarda y muestra la gráfica actual
def guardar_grafica(nombre):
    plt.savefig(CARPETA_RESULTADOS / nombre, dpi=120, bbox_inches="tight")
    plt.show()


# 1. Cargar las fotos
fotos_con_tapa = cargar_fotos(CARPETA_DATASET / "Botella_con_tapa")
fotos_sin_tapa = cargar_fotos(CARPETA_DATASET / "Botella_sin_tapa")

if len(fotos_con_tapa) == 0:
    raise SystemExit("No se encontraron fotos de botellas con tapa.")
if len(fotos_sin_tapa) == 0:
    raise SystemExit("No se encontraron fotos de botellas sin tapa.")

print("Fotos con tapa:", fotos_con_tapa.shape)
print("Fotos sin tapa:", fotos_sin_tapa.shape)

# 2. Dividir las fotos con tapa: 60 % entrenamiento, 20 % validación y 20 % prueba
entrenamiento_validacion, prueba = train_test_split(
    fotos_con_tapa, test_size=0.20, random_state=42, shuffle=True
)
entrenamiento, validacion = train_test_split(
    entrenamiento_validacion, test_size=0.25, random_state=42, shuffle=True
)

print("Entrenamiento:", entrenamiento.shape)
print("Validación:", validacion.shape)
print("Prueba:", prueba.shape)

# 3. Crear el modelo
autoencoder = crear_autoencoder()
autoencoder.summary()

# 4. Aumento de datos: posición, acercamiento, brillo y contraste
aumento = tf.keras.Sequential([
    layers.RandomTranslation(0.10, 0.25, fill_mode="nearest"),
    layers.RandomZoom(0.15, fill_mode="nearest"),
    layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
    layers.RandomContrast(0.15),
])


def aumentar(lote):
    lote_aumentado = tf.clip_by_value(aumento(lote, training=True), 0.0, 1.0)
    return lote_aumentado, lote_aumentado


datos = (
    tf.data.Dataset.from_tensor_slices(entrenamiento)
    .shuffle(len(entrenamiento), seed=42)
    .batch(LOTE)
    .map(aumentar)
)

# 5. Entrenar y detener si la validación deja de mejorar
parada_temprana = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)

historial = autoencoder.fit(
    datos,
    epochs=EPOCAS,
    validation_data=(validacion, validacion),
    callbacks=[parada_temprana],
)

# 6. Guardar el historial y la curva de entrenamiento
with open(CARPETA_RESULTADOS / "historial_entrenamiento.json", "w") as archivo:
    json.dump(historial.history, archivo)

plt.figure(figsize=(8, 5))
plt.plot(historial.history["loss"], label="Entrenamiento")
plt.plot(historial.history["val_loss"], label="Validación")
plt.xlabel("Época")
plt.ylabel("MAE")
plt.title("Curva de entrenamiento")
plt.legend()
plt.grid(True, alpha=0.3)
guardar_grafica("1_curva_entrenamiento.png")

# 7. Guardar el modelo
autoencoder.save(RUTA_MODELO)
print("Modelo guardado en:", RUTA_MODELO)

# 8. Calcular el umbral: promedio + 3 desviaciones estándar
mae_entrenamiento = calcular_mae(entrenamiento, autoencoder.predict(entrenamiento, verbose=0))
promedio = float(np.mean(mae_entrenamiento))
desviacion = float(np.std(mae_entrenamiento))
umbral = promedio + NUM_DESVIACIONES * desviacion

print(f"Promedio MAE: {promedio:.5f}")
print(f"Desviación estándar MAE: {desviacion:.5f}")
print(f"Umbral: {umbral:.5f}")

RUTA_UMBRAL.write_text(f"{umbral:.6f}\n")
print("Umbral guardado en:", RUTA_UMBRAL)

plt.figure(figsize=(10, 5))
plt.hist(mae_entrenamiento, bins=50, edgecolor="white")
plt.axvline(umbral, color="red", linestyle="--", label=f"Umbral: {umbral:.4f}")
plt.xlabel("MAE")
plt.ylabel("Número de imágenes")
plt.title("Distribución del error en entrenamiento")
plt.legend()
plt.grid(True, alpha=0.3)
guardar_grafica("2_histograma_umbral.png")

# 9. Mostrar ejemplos de reconstrucción
i_con_tapa = np.random.randint(0, len(prueba))
i_sin_tapa = np.random.randint(0, len(fotos_sin_tapa))
ejemplos = [
    ("Con tapa", prueba[i_con_tapa:i_con_tapa + 1]),
    ("Sin tapa", fotos_sin_tapa[i_sin_tapa:i_sin_tapa + 1]),
]

figura, ejes = plt.subplots(2, 2, figsize=(8, 8))
for fila, (nombre, imagen) in enumerate(ejemplos):
    reconstruccion = autoencoder.predict(imagen, verbose=0)
    mae = calcular_mae(imagen, reconstruccion)[0]
    es_anomalia = mae > umbral

    ejes[fila, 0].imshow(imagen[0])
    ejes[fila, 0].add_patch(plt.Rectangle(
        (COL_INICIO, FILA_INICIO), COL_FIN - COL_INICIO, FILA_FIN - FILA_INICIO - 1,
        fill=False, edgecolor="yellow", linewidth=2,
    ))
    ejes[fila, 0].set_title("Entrada: " + nombre)
    ejes[fila, 0].axis("off")

    ejes[fila, 1].imshow(reconstruccion[0])
    ejes[fila, 1].set_title(
        f"MAE: {mae:.5f}\n{'ANOMALIA' if es_anomalia else 'NO ANOMALIA'}",
        color="crimson" if es_anomalia else "darkgreen",
    )
    ejes[fila, 1].axis("off")

figura.suptitle(f"Umbral: {umbral:.5f}")
plt.tight_layout()
guardar_grafica("3_ejemplos_reconstruccion.png")

# 10. Evaluar el modelo
mae_con_tapa = calcular_mae(prueba, autoencoder.predict(prueba, verbose=0))
mae_sin_tapa = calcular_mae(fotos_sin_tapa, autoencoder.predict(fotos_sin_tapa, verbose=0))

clase_real = np.concatenate([np.zeros(len(mae_con_tapa)), np.ones(len(mae_sin_tapa))]).astype(int)
clase_predicha = (np.concatenate([mae_con_tapa, mae_sin_tapa]) > umbral).astype(int)
matriz = confusion_matrix(clase_real, clase_predicha)

print("\nResultados de la evaluación:")
print(f"MAE con tapa: promedio {mae_con_tapa.mean():.5f}, máximo {mae_con_tapa.max():.5f}")
print(f"MAE sin tapa: promedio {mae_sin_tapa.mean():.5f}, mínimo {mae_sin_tapa.min():.5f}")
print(f"Exactitud: {accuracy_score(clase_real, clase_predicha):.4f}")
print(f"Precisión: {precision_score(clase_real, clase_predicha, zero_division=0):.4f}")
print(f"Sensibilidad: {recall_score(clase_real, clase_predicha, zero_division=0):.4f}")
print("Matriz de confusión:", matriz.tolist())

# 11. Guardar la matriz de confusión
figura, eje = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(matriz, display_labels=["Con tapa", "Sin tapa"]).plot(
    cmap=plt.cm.Blues, ax=eje, values_format="d"
)
plt.title("Matriz de confusión")
plt.grid(False)
plt.tight_layout()
guardar_grafica("4_matriz_confusion.png")
