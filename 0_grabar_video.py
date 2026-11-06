"""Graba un video con la cámara. Presione q para terminar."""

import cv2

# Configuración
NUM_CAMARA = 1
NOMBRE_VIDEO = "video_con_sin_tapa_prueba_funcionamiento.mp4"
CUADROS_POR_SEGUNDO = 20.0

# Abrir la cámara y preparar el archivo de video
camara = cv2.VideoCapture(NUM_CAMARA)
if not camara.isOpened():
    raise SystemExit(f"No se pudo abrir la camara {NUM_CAMARA}")

ancho = int(camara.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(NOMBRE_VIDEO, codec, CUADROS_POR_SEGUNDO, (ancho, alto))

print("Grabando:", NOMBRE_VIDEO)
print("Presione q para terminar")

# Grabar hasta presionar q
cuadros = 0
while True:
    leido, imagen = camara.read()
    if not leido:
        print("No se recibio imagen de la camara")
        break

    # Voltear en espejo, igual que los videos de entrenamiento
    imagen = cv2.flip(imagen, 1)
    video.write(imagen)
    cuadros += 1

    cv2.imshow("Grabando - presione q para salir", imagen)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cerrar la cámara y guardar el video
camara.release()
video.release()
cv2.destroyAllWindows()

print("Video guardado:", NOMBRE_VIDEO)
print("Cuadros grabados:", cuadros)
