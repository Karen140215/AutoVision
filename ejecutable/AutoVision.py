"""AutoVision v1.0 - versión ejecutable de 4_probar_con_camara.py.

Al abrir, una ventana pregunta si se usa el video de prueba o la cámara
(o se indica al iniciar: AutoVision.exe --video  /  --video ruta  /  --camara 0).
En la ventana de resultados, presione q o ESC (o cierre la ventana) para salir.

Funciona en Windows y en Linux (por ejemplo, Raspberry Pi OS de 64 bits).
En Linux se ejecuta con Python: python3 ejecutable/AutoVision.py
"""

import ctypes
import gc
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Sin ventana de consola, Python no tiene dónde escribir los mensajes
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

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

EN_WINDOWS = sys.platform == "win32"
FUENTE_UI = "Segoe UI" if EN_WINDOWS else "DejaVu Sans"

# Carpeta del programa: junto al .exe, o junto a este archivo si se corre con Python.
# En el repositorio este archivo está en ejecutable/ y el modelo en la carpeta de arriba.
if getattr(sys, "frozen", False):
    CARPETA = Path(sys.executable).resolve().parent
else:
    CARPETA = Path(__file__).resolve().parent
    if not (CARPETA / "modelo_entrenado").exists() and (CARPETA.parent / "modelo_entrenado").exists():
        CARPETA = CARPETA.parent

# Rutas relativas: OpenCV falla con rutas que tienen tildes o eñes,
# así que se trabaja desde la carpeta del programa
os.chdir(CARPETA)
RUTA_MODELO = Path("modelo_entrenado") / "autoencoder_botellas.keras"
RUTA_UMBRAL = Path("modelo_entrenado") / "umbral.txt"
RUTA_VIDEO = Path("videos") / "prueba" / NOMBRE_VIDEO

VENTANA = "AutoVision v1.0 - Control de calidad"

# Paleta oscura (la misma en todas las ventanas)
FONDO = "#1E2127"
TARJETA = "#272B33"
TARJETA_ACTIVA = "#323743"
LINEA = "#3E434D"
TEXTO = "#ECEFF4"
TEXTO_SUAVE = "#AAB1BD"
MORADO = "#8B5CF6"
MORADO_CLARO = "#A78BFA"
ROJO = "#E5484D"


# Evita que Windows amplíe las ventanas y las vea borrosas
def activar_alta_resolucion():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def escala_pantalla():
    try:
        return ctypes.windll.user32.GetDpiForSystem() / 96
    except Exception:
        return 1.0


def tamano_pantalla():
    try:
        return ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
    except Exception:
        pass
    try:
        return base().winfo_screenwidth(), base().winfo_screenheight()
    except Exception:
        return 1920, 1080


# Pone oscura la barra de título de una ventana (Windows 10 y 11)
def titulo_oscuro(hwnd):
    try:
        if hwnd:
            valor = ctypes.c_int(1)
            for atributo in (20, 19):
                if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, atributo, ctypes.byref(valor), 4) == 0:
                    break
    except Exception:
        pass


# Ubica la ventana en el centro de la pantalla y la pone al frente
def centrar(ventana):
    ventana.withdraw()
    ventana.update_idletasks()
    if EN_WINDOWS:
        titulo_oscuro(ctypes.windll.user32.GetParent(ventana.winfo_id()))
    ancho, alto = ventana.winfo_reqwidth(), ventana.winfo_reqheight()
    x = (ventana.winfo_screenwidth() - ancho) // 2
    y = (ventana.winfo_screenheight() - alto) // 2
    ventana.geometry(f"+{x}+{y}")
    ventana.deiconify()
    ventana.update_idletasks()

    def quitar_encima():
        if ventana.winfo_exists():
            ventana.attributes("-topmost", False)

    def al_frente():
        if not ventana.winfo_exists():
            return
        ventana.lift()
        ventana.attributes("-topmost", True)
        base().after(300, quitar_encima)
        ventana.focus_force()

    base().after(50, al_frente)


# Tk debe crearse y borrarse siempre en el hilo principal. Por eso hay una
# sola base, que vive hasta el final, y cada ventana es una Toplevel.
_BASE = None


def base():
    global _BASE
    if _BASE is None:
        _BASE = tk.Tk()
        _BASE.withdraw()
    return _BASE


# Espera a que se cierre la ventana y limpia la memoria en el hilo principal
def esperar(ventana):
    ventana.wait_window()
    gc.collect()


def nueva_ventana():
    raiz = tk.Toplevel(base())
    raiz.title("AutoVision v1.0")
    raiz.resizable(False, False)
    raiz.configure(bg=FONDO)
    return raiz


# Botón plano con borde de color
def boton(padre, texto, comando, color=MORADO, relleno=TARJETA):
    borde = tk.Frame(padre, bg=color, padx=1, pady=1)
    b = tk.Button(
        borde, text=texto, command=comando, font=(FUENTE_UI, 12, "bold"),
        bg=relleno, fg=TEXTO, activebackground=TARJETA_ACTIVA, activeforeground=TEXTO,
        relief="flat", bd=0, cursor="hand2", padx=16, pady=9,
    )
    b.pack(fill="both", expand=True)
    b.bind("<Enter>", lambda e: b.configure(bg=TARJETA_ACTIVA))
    b.bind("<Leave>", lambda e: b.configure(bg=relleno))
    return borde


# Aviso de error con el mismo estilo oscuro
def error(mensaje):
    print("ERROR:", mensaje)
    raiz = nueva_ventana()
    tk.Label(raiz, text="No se pudo continuar", font=(FUENTE_UI, 14, "bold"),
             bg=FONDO, fg=ROJO).pack(padx=36, pady=(24, 6))
    tk.Label(raiz, text=mensaje, font=(FUENTE_UI, 10), bg=FONDO, fg=TEXTO,
             justify="center", wraplength=420).pack(padx=36)
    boton(raiz, "Aceptar", raiz.destroy, color=ROJO).pack(pady=(20, 24), ipadx=20)
    raiz.bind("<Return>", lambda e: raiz.destroy())
    raiz.bind("<Escape>", lambda e: raiz.destroy())
    centrar(raiz)
    esperar(raiz)
    salir(1)


MAX_CAMARAS = 5
MINIATURA = (192, 144)


# Forma de abrir las cámaras: en Windows, DirectShow (abre en menos de un
# segundo); en Linux, V4L2. Si falla, el modo predeterminado de OpenCV.
APIS_CAMARA = (cv2.CAP_DSHOW, cv2.CAP_ANY) if EN_WINDOWS else (cv2.CAP_V4L2, cv2.CAP_ANY)

# Dispositivos internos de la Raspberry Pi que aparecen como /dev/video
# pero no son cámaras
NO_CAMARAS = ("bcm2835-codec", "bcm2835-isp", "rpi-hevc", "pispbe", "rp1-cfe", "hevc")


def abrir_camara(num):
    for api in APIS_CAMARA:
        camara = cv2.VideoCapture(num, api)
        if camara.isOpened():
            return camara
        camara.release()
    return None


# Números de cámara que vale la pena probar
def numeros_camaras():
    if EN_WINDOWS:
        return list(range(MAX_CAMARAS))
    numeros = []
    for disp in Path("/sys/class/video4linux").glob("video*"):
        try:
            num = int(disp.name[5:])
            nombre = (disp / "name").read_text().strip().lower()
        except (ValueError, OSError):
            continue
        if not any(n in nombre for n in NO_CAMARAS):
            numeros.append(num)
    return sorted(numeros) or list(range(MAX_CAMARAS))


# Busca las cámaras conectadas y toma una imagen de cada una
def detectar_camaras():
    encontradas = []
    for num in numeros_camaras():
        camara = cv2.VideoCapture(num, APIS_CAMARA[0])
        if not camara.isOpened():
            camara.release()
            continue
        imagen = None
        for _ in range(8):  # las primeras imágenes pueden salir oscuras
            leido, cuadro = camara.read()
            if leido:
                imagen = cuadro
        camara.release()
        if imagen is not None:
            if ESPEJO:
                imagen = cv2.flip(imagen, 1)
            imagen = cv2.cvtColor(cv2.resize(imagen, MINIATURA), cv2.COLOR_BGR2RGB)
            encontradas.append((num, imagen))
    return encontradas


# Ventana inicial: elegir el video de prueba o una de las cámaras detectadas
def elegir_entrada():
    eleccion = {}
    raiz = nueva_ventana()
    miniaturas = []  # Tk necesita conservar las imágenes

    tk.Label(raiz, text="AutoVision v1.0", font=(FUENTE_UI, 22, "bold"),
             bg=FONDO, fg=TEXTO).pack(padx=48, pady=(26, 0))
    tk.Label(raiz, text="Detección de botellas sin tapa con autoencoder",
             font=(FUENTE_UI, 10), bg=FONDO, fg=TEXTO_SUAVE).pack(padx=48, pady=(2, 0))
    tk.Frame(raiz, bg=MORADO, height=2, width=60).pack(pady=(14, 18))

    cuerpo = tk.Frame(raiz, bg=FONDO)
    cuerpo.pack(fill="both", padx=40)
    tk.Label(cuerpo, text="¿Qué entrada desea usar?", font=(FUENTE_UI, 12),
             bg=FONDO, fg=TEXTO).pack(pady=(0, 12))

    def usar(tipo, num=0):
        eleccion["tipo"] = tipo
        eleccion["camara"] = num
        raiz.destroy()

    def elegir_archivo():
        ruta = filedialog.askopenfilename(
            parent=raiz, title="Elegir un video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.MP4 *.AVI *.MOV *.MKV *.WMV *.M4V"),
                       ("Todos los archivos", "*")],
        )
        if ruta:
            eleccion["ruta"] = ruta
            usar("video")

    fila_videos = tk.Frame(cuerpo, bg=FONDO)
    fila_videos.pack(fill="x")
    boton(fila_videos, "Video de prueba", lambda: usar("video")).pack(side="left", fill="x", expand=True, padx=(0, 5))
    boton(fila_videos, "Abrir un video...", elegir_archivo).pack(side="left", fill="x", expand=True, padx=(5, 0))

    encabezado = tk.Frame(cuerpo, bg=FONDO)
    encabezado.pack(fill="x", pady=(18, 8))
    tk.Label(encabezado, text="Cámaras conectadas", font=(FUENTE_UI, 11, "bold"),
             bg=FONDO, fg=TEXTO).pack(side="left")
    buscar = tk.Label(encabezado, text="Buscar de nuevo", font=(FUENTE_UI, 9, "underline"),
                      bg=FONDO, fg=MORADO_CLARO, cursor="hand2")
    buscar.pack(side="right")

    zona = tk.Frame(cuerpo, bg=FONDO)
    zona.pack(fill="x")

    tk.Label(raiz, text="Haga clic en la imagen de la cámara que desea usar.\nPara salir: q, ESC o cerrar la ventana.",
             font=(FUENTE_UI, 9), bg=FONDO, fg=TEXTO_SUAVE, justify="center").pack(pady=(14, 24))

    def tarjeta(padre, num, imagen):
        borde = tk.Frame(padre, bg=LINEA, padx=1, pady=1, cursor="hand2")
        interior = tk.Frame(borde, bg=TARJETA, cursor="hand2")
        interior.pack()
        foto = ImageTk.PhotoImage(Image.fromarray(imagen))
        miniaturas.append(foto)
        tk.Label(interior, image=foto, bg=TARJETA, cursor="hand2").pack(padx=8, pady=(8, 4))
        if EN_WINDOWS:
            nota = "Suele ser la del computador" if num == 0 else "Suele ser una cámara USB"
        else:
            nota = f"/dev/video{num}"
        tk.Label(interior, text=f"Cámara {num}", font=(FUENTE_UI, 10, "bold"), bg=TARJETA, fg=TEXTO,
                 cursor="hand2").pack()
        tk.Label(interior, text=nota, font=(FUENTE_UI, 8), bg=TARJETA, fg=TEXTO_SUAVE,
                 cursor="hand2").pack(pady=(0, 8))
        partes = [borde, interior] + list(interior.winfo_children())
        for w in partes:
            w.bind("<Button-1>", lambda e: usar("camara", num))
            w.bind("<Enter>", lambda e: borde.configure(bg=MORADO))
            w.bind("<Leave>", lambda e: borde.configure(bg=LINEA))
        return borde

    def mostrar(camaras):
        for w in zona.winfo_children():
            w.destroy()
        miniaturas.clear()
        if not camaras:
            tk.Label(zona, text="No se encontró ninguna cámara.\nConéctela y haga clic en «Buscar de nuevo».",
                     font=(FUENTE_UI, 10), bg=FONDO, fg=TEXTO_SUAVE, justify="center").pack(pady=10)
        else:
            fila = None
            for i, (num, imagen) in enumerate(camaras):
                if i % 3 == 0:
                    fila = tk.Frame(zona, bg=FONDO)
                    fila.pack(pady=(0, 10))
                tarjeta(fila, num, imagen).pack(side="left", padx=5)
        buscar.bind("<Button-1>", lambda e: iniciar_busqueda())
        recentrar()

    def iniciar_busqueda():
        buscar.unbind("<Button-1>")
        for w in zona.winfo_children():
            w.destroy()
        tk.Label(zona, text="Buscando cámaras...", font=(FUENTE_UI, 10),
                 bg=FONDO, fg=TEXTO_SUAVE).pack(pady=10)
        resultado = {}

        def tarea():
            try:
                resultado["camaras"] = detectar_camaras()
            except Exception:
                resultado["camaras"] = []

        hilo = threading.Thread(target=tarea, daemon=True)
        hilo.start()

        def revisar():
            if not raiz.winfo_exists():
                return
            if hilo.is_alive():
                base().after(100, revisar)
            else:
                mostrar(resultado["camaras"])

        base().after(100, revisar)

    def recentrar():
        raiz.update_idletasks()
        ancho, alto = raiz.winfo_reqwidth(), raiz.winfo_reqheight()
        x = (raiz.winfo_screenwidth() - ancho) // 2
        y = (raiz.winfo_screenheight() - alto) // 2
        raiz.geometry(f"+{x}+{y}")

    raiz.bind("<Escape>", lambda e: raiz.destroy())
    centrar(raiz)
    base().after(200, lambda: raiz.winfo_exists() and iniciar_busqueda())
    esperar(raiz)
    return eleccion


class ErrorEntrada(Exception):
    pass


# Abre el video o la cámara (se llama desde la ventana de carga)
# OpenCV no abre rutas con tildes o eñes. Se prueba la ruta normal, luego la
# ruta corta de Windows y, por último, una copia temporal con nombre simple.
def abrir_video(ruta):
    entrada = cv2.VideoCapture(str(ruta))
    if entrada.isOpened():
        return entrada
    entrada.release()
    try:
        corta = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(str(ruta), corta, 1024):
            entrada = cv2.VideoCapture(corta.value)
            if entrada.isOpened():
                return entrada
            entrada.release()
    except Exception:
        pass
    try:
        copia = Path(tempfile.gettempdir()) / f"AutoVision_video{ruta.suffix.lower()}"
        shutil.copyfile(ruta, copia)
        entrada = cv2.VideoCapture(str(copia))
        if entrada.isOpened():
            return entrada
        entrada.release()
    except Exception:
        pass
    return None


def nombre_corto(texto, maximo=34):
    return texto if len(texto) <= maximo else texto[: maximo - 3] + "..."


def abrir_entrada(eleccion):
    if eleccion["tipo"] == "video" and eleccion.get("ruta"):
        ruta = Path(eleccion["ruta"])
        if not ruta.exists():
            raise ErrorEntrada(f"No se encontró el video:\n{ruta}")
        entrada = abrir_video(ruta)
        if entrada is None:
            raise ErrorEntrada(f"No se pudo abrir el video:\n{ruta.name}\n\nPruebe con un archivo .mp4 o .avi.")
        leido, _cuadro = entrada.read()
        if not leido:
            entrada.release()
            raise ErrorEntrada(f"El video no tiene imágenes que se puedan leer:\n{ruta.name}")
        entrada.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print("Usando video:", ruta)
        return entrada, f"video {nombre_corto(ruta.name)}"

    if eleccion["tipo"] == "video":
        if not RUTA_VIDEO.exists():
            raise ErrorEntrada(f"No se encontró el video:\n{CARPETA / RUTA_VIDEO}")
        entrada = cv2.VideoCapture(str(RUTA_VIDEO))
        if not entrada.isOpened():
            raise ErrorEntrada(f"No se pudo abrir el video:\n{CARPETA / RUTA_VIDEO}")
        print("Usando video:", NOMBRE_VIDEO)
        return entrada, "cámara simulada (video de prueba)"

    num = eleccion["camara"]
    entrada = abrir_camara(num)
    if entrada is None:
        raise ErrorEntrada(f"No se pudo abrir la cámara {num}.\nRevise que esté conectada y que otro programa no la esté usando.")
    entrada.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAM)
    entrada.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAM)
    for _ in range(30):  # esperar la primera imagen
        leido, _cuadro = entrada.read()
        if leido:
            break
    else:
        entrada.release()
        raise ErrorEntrada(f"La cámara {num} abrió, pero no envía imagen.\nPruebe con otra cámara.")
    print("Usando camara", num)
    return entrada, f"cámara {num}"


# Carga el modelo y abre la entrada mostrando una ventana de espera.
# La ventana se cierra cuando todo está listo para mostrar los resultados.
def preparar_todo(eleccion):
    resultado = {}
    paso = {"texto": "Cargando el modelo..."}

    def tarea():
        try:
            import tensorflow as tf
            resultado["modelo"] = tf.keras.models.load_model(RUTA_MODELO)
            if eleccion["tipo"] == "video":
                paso["texto"] = "Abriendo el video..." if eleccion.get("ruta") else "Abriendo el video de prueba..."
            else:
                paso["texto"] = f"Abriendo la cámara {eleccion['camara']}..."
            resultado["entrada"], resultado["titulo"] = abrir_entrada(eleccion)
        except ErrorEntrada as e:
            resultado["error"] = str(e)
        except Exception as e:
            resultado["error"] = f"No se pudo cargar el modelo:\n{e}"

    raiz = nueva_ventana()
    raiz.protocol("WM_DELETE_WINDOW", lambda: None)
    estilo = ttk.Style(raiz)
    estilo.theme_use("clam")
    estilo.configure("Morado.Horizontal.TProgressbar", troughcolor=TARJETA, background=MORADO,
                     bordercolor=FONDO, lightcolor=MORADO, darkcolor=MORADO, thickness=6)
    titulo = tk.Label(raiz, text=paso["texto"], font=(FUENTE_UI, 14, "bold"), bg=FONDO, fg=TEXTO, width=28)
    titulo.pack(padx=56, pady=(28, 4))
    tk.Label(raiz, text="Esto puede tardar unos segundos.", font=(FUENTE_UI, 10),
             bg=FONDO, fg=TEXTO_SUAVE).pack(padx=56)
    barra = ttk.Progressbar(raiz, mode="indeterminate", length=300, style="Morado.Horizontal.TProgressbar")
    barra.pack(padx=56, pady=(18, 30))
    barra.start(12)
    centrar(raiz)

    hilo = threading.Thread(target=tarea, daemon=True)
    hilo.start()

    def revisar():
        if titulo.cget("text") != paso["texto"]:
            titulo.configure(text=paso["texto"])
        if hilo.is_alive():
            raiz.after(100, revisar)
        else:
            raiz.destroy()

    raiz.after(100, revisar)
    esperar(raiz)

    if "error" in resultado:
        error(resultado["error"])
    return resultado["modelo"], resultado["entrada"], resultado["titulo"]


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


# ---------- Ventana de resultados ----------

# Medidas base (se multiplican por la escala de la pantalla)
PANEL = 400        # lado de cada imagen
MARGEN = 16        # borde exterior y separación entre imágenes
TITULO = 34        # alto de la fila de títulos
ESTADO_ANCHO, ESTADO_ALTO = 250, 56
METRICAS = 46      # alto de la fila inferior
BASE_ANCHO = 2 * PANEL + 3 * MARGEN
BASE_ALTO = 12 + TITULO + PANEL + MARGEN + ESTADO_ALTO + 14 + METRICAS + 6

# Colores (RGB)
C_FONDO = (30, 33, 39)
C_TEXTO = (236, 239, 244)
C_TEXTO_SUAVE = (170, 177, 189)
C_LINEA = (62, 67, 77)
C_VERDE = (76, 175, 80)
C_ROJO = (229, 72, 77)
C_ACENTO = (167, 139, 250)
C_AMARILLO = (255, 214, 0)

ESCALA_VENTANA = 1.0
CARPETA_FUENTES = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
DEJAVU = Path("/usr/share/fonts/truetype/dejavu")
# Letras que se prueban en orden: Segoe UI (Windows) y DejaVu Sans (Linux)
ARCHIVOS_FUENTE = {
    False: [CARPETA_FUENTES / "segoeui.ttf", DEJAVU / "DejaVuSans.ttf", "DejaVuSans.ttf"],
    True: [CARPETA_FUENTES / "segoeuib.ttf", DEJAVU / "DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"],
}
_fuentes = {}
_mascara = {}


def fuente(tamano, negrita=False):
    tamano = max(8, int(round(tamano * ESCALA_VENTANA)))
    clave = (tamano, negrita)
    if clave not in _fuentes:
        for archivo in ARCHIVOS_FUENTE[negrita]:
            try:
                _fuentes[clave] = ImageFont.truetype(str(archivo), tamano)
                break
            except OSError:
                continue
        else:
            try:
                _fuentes[clave] = ImageFont.load_default(tamano)
            except TypeError:  # Pillow anterior a 10.1
                _fuentes[clave] = ImageFont.load_default()
    return _fuentes[clave]


def u(valor):
    return int(round(valor * ESCALA_VENTANA))


# Máscara con esquinas redondeadas para las imágenes
def mascara(lado):
    if lado not in _mascara:
        m = Image.new("L", (lado, lado), 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, lado - 1, lado - 1], radius=u(6), fill=255)
        _mascara[lado] = m
    return _mascara[lado]


# Une la imagen, la reconstrucción y los resultados
def armar_pantalla(cuadro, reconstruccion, mae, umbral, fps, inferencia, titulo):
    p, m = u(PANEL), u(MARGEN)
    ancho, alto = u(BASE_ANCHO), u(BASE_ALTO)
    x_izq, x_der = m, 2 * m + p
    y_titulo = u(12)
    y_img = y_titulo + u(TITULO)

    entrada = cv2.cvtColor(cv2.resize(cuadro, (p, p)), cv2.COLOR_BGR2RGB)
    salida = np.clip(reconstruccion[0] * 255, 0, 255).astype(np.uint8)
    salida = cv2.resize(salida, (p, p))

    lienzo = Image.new("RGB", (ancho, alto), C_FONDO)
    lienzo.paste(Image.fromarray(entrada), (x_izq, y_img), mascara(p))
    lienzo.paste(Image.fromarray(salida), (x_der, y_img), mascara(p))
    d = ImageDraw.Draw(lienzo)

    # Títulos
    d.text((x_izq + u(4), y_titulo + u(TITULO) // 2 - u(2)), f"Entrada: {titulo}",
           font=fuente(15, True), fill=C_TEXTO, anchor="lm")
    d.text((x_der + u(4), y_titulo + u(TITULO) // 2 - u(2)), "Reconstrucción del modelo",
           font=fuente(15, True), fill=C_TEXTO, anchor="lm")

    # Zona donde se mide el error
    esc = p / ANCHO
    x1, y1 = x_izq + int(COL_INICIO * esc), y_img + int(FILA_INICIO * esc)
    x2, y2 = x_izq + int(COL_FIN * esc) - 1, y_img + int(FILA_FIN * esc) - 1
    d.rectangle([x1, y1, x2, y2], outline=C_AMARILLO, width=max(2, u(2)))
    etiqueta = "Zona evaluada"
    fe = fuente(12, True)
    d.rectangle([x1, y1 - u(20), x1 + int(d.textlength(etiqueta, font=fe)) + u(12), y1], fill=C_AMARILLO)
    d.text((x1 + u(6), y1 - u(10)), etiqueta, font=fe, fill=(20, 20, 20), anchor="lm")

    # Estado, centrado
    es_anomalia = mae > umbral
    color = C_ROJO if es_anomalia else C_VERDE
    estado = "ANOMALÍA" if es_anomalia else "NO ANOMALÍA"
    detalle = "El MAE supera el umbral" if es_anomalia else "El MAE no supera el umbral"
    ye = y_img + p + m
    ea, eh = u(ESTADO_ANCHO), u(ESTADO_ALTO)
    ex = (ancho - ea) // 2
    d.rounded_rectangle([ex, ye, ex + ea, ye + eh], radius=u(8), outline=color, width=max(1, u(1.5)))
    d.text((ancho // 2, ye + u(20)), estado, font=fuente(20, True), fill=color, anchor="mm")
    d.text((ancho // 2, ye + u(42)), detalle, font=fuente(12), fill=C_TEXTO, anchor="mm")

    # Línea y fila de datos
    yl = ye + eh + u(14)
    d.line([x_izq, yl, ancho - m, yl], fill=C_LINEA, width=1)
    ym = yl + u(METRICAS) // 2
    datos = [
        ("MAE", f"{mae:.4f}", color),
        ("Umbral", f"{umbral:.4f}", C_ACENTO),
        ("Inferencia", f"{inferencia:.1f} ms", C_ACENTO),
        ("FPS", f"{fps:.1f}", C_ACENTO),
    ]
    columna = (ancho - 2 * m) / len(datos)
    fl, fv = fuente(13), fuente(13)
    for i, (nombre, valor, color_valor) in enumerate(datos):
        cx = int(m + columna * (i + 0.5))
        texto = f"{nombre}: "
        total = d.textlength(texto, font=fl) + d.textlength(valor, font=fv)
        x = cx - total / 2
        d.text((x, ym), texto, font=fl, fill=C_TEXTO_SUAVE, anchor="lm")
        d.text((x + d.textlength(texto, font=fl), ym), valor, font=fv, fill=color_valor, anchor="lm")
        if i:
            xs = int(m + columna * i)
            d.line([xs, ym - u(11), xs, ym + u(11)], fill=C_LINEA, width=1)

    return cv2.cvtColor(np.asarray(lienzo), cv2.COLOR_RGB2BGR)


def barra_oscura():
    if EN_WINDOWS:
        titulo_oscuro(ctypes.windll.user32.FindWindowW(None, VENTANA))


# Crea la ventana de resultados, con el tamaño de la pantalla y en el centro
def preparar_ventana():
    global ESCALA_VENTANA
    pantalla_ancho, pantalla_alto = tamano_pantalla()
    ESCALA_VENTANA = min(escala_pantalla(), 0.85 * pantalla_alto / BASE_ALTO, 0.9 * pantalla_ancho / BASE_ANCHO)
    ancho, alto = u(BASE_ANCHO), u(BASE_ALTO)
    # GUI_NORMAL quita la barra de herramientas que OpenCV agrega en Linux
    cv2.namedWindow(VENTANA, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow(VENTANA, max(0, (pantalla_ancho - ancho) // 2), max(0, (pantalla_alto - alto) // 2 - u(30)))


def leer_argumentos():
    # Opcional: AutoVision.exe --video  /  --video "C:\ruta\video.mp4"  /  --camara 1
    args = sys.argv[1:]
    if "--video" in args:
        i = args.index("--video")
        eleccion = {"tipo": "video", "camara": 0}
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            eleccion["ruta"] = args[i + 1]
        return eleccion
    if "--camara" in args:
        i = args.index("--camara")
        num = int(args[i + 1]) if i + 1 < len(args) and args[i + 1].isdigit() else 0
        return {"tipo": "camara", "camara": num}
    return None


def main():
    activar_alta_resolucion()
    eleccion = leer_argumentos() or elegir_entrada()
    if not eleccion:
        return
    usar_video = eleccion["tipo"] == "video"

    # Cargar el modelo y abrir la entrada (con ventana de espera)
    if not RUTA_MODELO.exists() or not RUTA_UMBRAL.exists():
        error(f"No se encontró la carpeta modelo_entrenado junto al programa:\n{CARPETA}")
    umbral = float(RUTA_UMBRAL.read_text())
    print(f"Umbral: {umbral:.5f}")
    modelo, entrada, titulo = preparar_todo(eleccion)
    fps_video = entrada.get(cv2.CAP_PROP_FPS) or 20
    print("Presione q o ESC para salir")

    preparar_ventana()
    primera = True
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

        # La primera vez, traer la ventana al frente
        if primera:
            barra_oscura()
            try:
                cv2.setWindowProperty(VENTANA, cv2.WND_PROP_TOPMOST, 1)
                cv2.waitKey(1)
                cv2.setWindowProperty(VENTANA, cv2.WND_PROP_TOPMOST, 0)
            except cv2.error:
                pass
            primera = False

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


def salir(codigo=0):
    global _BASE
    if _BASE is not None:
        try:
            _BASE.destroy()
        except tk.TclError:
            pass
        _BASE = None
    gc.collect()
    sys.exit(codigo)


if __name__ == "__main__":
    main()
    salir(0)
