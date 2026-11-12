# AutoVision v1.0

Detección de botellas PET sin tapa mediante un autoencoder convolucional y visión artificial.

Software desarrollado en el semillero de investigación TURING, Grupo de Investigación en Aplicaciones Mecatrónicas (GRAM), Facultad de Ingeniería Mecatrónica, Universidad Santo Tomás, Seccional Bucaramanga.

## Autores

- Karen Johanna García Martínez
- María Elizabeth Genes García
- Samuel Joel García Roa
- Edward Alexander Espinel Guauque
- Pablo Alexander Hernández Achagua
- Oscar Fernando Becerra Angarita (docente asesor)

## Descargar la versión ejecutable para Windows

No requiere instalar Python ni Anaconda.

1. Descargar `AutoVision_v1.0_Windows.zip` desde la sección [Releases](https://github.com/Karen140215/AutoVision/releases/latest) (cerca de 470 MB).
2. Hacer clic derecho sobre el archivo, elegir «Extraer todo» y abrir `AutoVision.exe`.
3. Elegir «Video de prueba» o «Cámara».

El programa no tiene firma digital, por lo que Windows puede mostrar «Windows protegió su PC». En ese caso se elige «Más información» y luego «Ejecutar de todas formas». El manual de usuario de la versión ejecutable también está en la sección Releases.

## Cómo funciona

El autoencoder se entrenó solo con imágenes de botellas con tapa. Cada imagen se reduce a 64 valores y se reconstruye. Si el error de reconstrucción (MAE) en la zona central supera el umbral de 0,045147, el programa muestra `ANOMALIA`. En caso contrario, muestra `NO ANOMALIA`.

## Contenido del repositorio

| Archivo o carpeta | Contenido |
| --- | --- |
| `0_grabar_video.py` | Graba videos con la cámara |
| `1_crear_dataset.py` | Extrae las imágenes de los videos de entrenamiento |
| `2_entrenar_autoencoder.py` | Entrena el modelo y calcula el umbral |
| `3_probar_con_video.py` | Analiza un video grabado y guarda los resultados |
| `4_probar_con_camara.py` | Analiza la imagen de la cámara en tiempo real |
| `modelo_entrenado/` | Modelo `autoencoder_botellas.keras` y `umbral.txt` |
| `videos/entrenamiento/` | Cinco videos usados para crear el dataset |
| `videos/prueba/` | Video de prueba con botellas con tapa y sin tapa |
| `dataset_botellas/` | 600 imágenes con tapa y 400 sin tapa |
| `resultados/` | Gráficas, registro de entrenamiento y video de resultado |
| `graficos_modelo/` | Gráficas explicativas del modelo |
| `requirements.txt` | Librerías y versiones probadas |
| `ejecutable/` | Código de la versión ejecutable (`AutoVision.py`) y archivo para generarla con PyInstaller (`AutoVision.spec`) |

## Instalación

Versiones probadas en Windows 11 con Python 3.10 y Anaconda. La carpeta debe estar en una ruta sin tildes ni eñes, porque OpenCV no lee imágenes en esas rutas.

```bash
conda create -n py310_prueba python=3.10
conda activate py310_prueba
conda install spyder
pip install -r requirements.txt
```

## Uso

### Probar el modelo con la cámara

1. Abrir Spyder desde el entorno `py310_prueba`.
2. Abrir `4_probar_con_camara.py` y asignar `USAR_VIDEO = False`. Con `True`, el programa simula la cámara con el video de prueba.
3. Ejecutar con F5.
4. Ubicar la botella dentro del rectángulo amarillo y leer el estado.
5. Hacer clic en la ventana y presionar `q` o `ESC` para salir.

El procedimiento completo de este programa está en el manual de usuario de AutoVision v1.0.

### Usar AutoVision en una Raspberry Pi

`AutoVision.exe` solo funciona en Windows. En una Raspberry Pi 4 (4 GB o más) o Pi 5 con Raspberry Pi OS de 64 bits, monitor y cámara USB, se ejecuta el código en Python:

```bash
sudo apt update
sudo apt install -y git python3-venv python3-tk
git clone https://github.com/Karen140215/AutoVision.git
cd AutoVision
python3 -m venv venv
source venv/bin/activate
pip install tensorflow opencv-python pillow numpy
python3 ejecutable/AutoVision.py
```

Se abre la misma ventana que en Windows, con el video de prueba y las cámaras conectadas. Para las veces siguientes basta con `cd AutoVision`, `source venv/bin/activate` y `python3 ejecutable/AutoVision.py`. También se puede iniciar directamente con `--video` o `--camara 0`. No se usa `requirements.txt`, porque sus versiones son las probadas en Windows con Python 3.10.

### Reentrenar el modelo

| Paso | Programa | Resultado |
| --- | --- | --- |
| 1 | `0_grabar_video.py` | Video nuevo, que se renombra y se mueve a `videos/entrenamiento` o `videos/prueba` |
| 2 | `1_crear_dataset.py` | Imágenes en `dataset_botellas/` (cerca de 1 minuto) |
| 3 | `2_entrenar_autoencoder.py` | Modelo, umbral y gráficas nuevos (cerca de 15 minutos) |
| 4 | `3_probar_con_video.py` | Video de resultado y MAE de cada cuadro en `resultados/` |

El reentrenamiento reemplaza el modelo y el umbral vigentes, por lo que se recomienda guardar antes una copia de `modelo_entrenado/` y `resultados/`. Los dos archivos de `modelo_entrenado/` deben provenir del mismo entrenamiento.

## Resultados

Evaluación con 120 imágenes con tapa y 400 sin tapa: 118 verdaderos negativos, 2 falsos positivos, 18 falsos negativos y 382 verdaderos positivos.

| Métrica | Valor |
| --- | --- |
| Exactitud | 96,2 % |
| Precisión | 99,5 % |
| Sensibilidad | 95,5 % |
| Especificidad | 98,3 % |
| F1-score | 97,4 % |

## Limitaciones

- Solo se ha validado la detección de botellas sin tapa.
- El resultado depende de que se mantengan la cámara, el fondo, la iluminación y la posición de los videos de entrenamiento.
- El programa no detecta si hay una botella; el estado solo debe leerse con la botella dentro del rectángulo amarillo.
- Las imágenes de evaluación provienen de los mismos videos que las de entrenamiento, lo que puede sobreestimar las métricas.

## Derechos

Este software es resultado del semillero de investigación TURING de la Universidad Santo Tomás, Seccional Bucaramanga. Todos los derechos reservados. Queda prohibida su reproducción total o parcial sin autorización de sus titulares.
