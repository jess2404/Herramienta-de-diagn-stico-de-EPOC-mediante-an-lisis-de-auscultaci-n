import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog # Agregado filedialog
from PIL import Image, ImageTk
import csv
import sounddevice as sd
from scipy.io.wavfile import write, read
from scipy.signal import spectrogram, windows
from sklearn.naive_bayes import GaussianNB
import datetime
import matplotlib.pyplot as plt 


grabando = False
stream = None
audio_buffer = []
fs_grabacion = 44100 # Frecuencia de muestreo
BASE_EPOC = "frecuencias_epoc"
BASE_SANOS = "frecuencias_sanos"
os.makedirs(BASE_EPOC, exist_ok=True)
os.makedirs(BASE_SANOS, exist_ok=True)

#Clasificación

def seleccionar_archivos(titulo):
    """Abre la ventana de diálogo para seleccionar archivos WAV."""
    archivos = filedialog.askopenfilenames(
        title=titulo, 
        filetypes=[("WAV files", "*.wav")]
    )
    # No es necesario crear/destruir la ventana Tk() aquí si ya estamos en un mainloop de Tkinter
    return list(archivos)

# ==========================================================
# 2. FUNCIONES DEL MOTOR DE CLASIFICACIÓN (CÓDIGO 1)
# ==========================================================

# ==========================================================
# 2. FUNCIONES DEL MOTOR DE CLASIFICACIÓN (CÓDIGO 1)
# ==========================================================
def mostrar_espectrograma(senal, fm, clasificacion):
    """
    Calcula y muestra el espectrograma con el resultado de la clasificación en el título.
    """
    duracion_deseada = 20  
    num_muestras = duracion_deseada * fm
    
    if len(senal) > num_muestras:
        senal = senal[:num_muestras]
    
    # Espectro
    frec, tiempo, espectro = spectrogram(
        senal,
        fs=fm,
        window=windows.hamming(512),
        nperseg=512,
        noverlap=256,
        nfft=1024,
        mode="magnitude"
    )

    # Gráfico con el título actualizado
    plt.figure()
    plt.pcolormesh(tiempo, frec, 20*np.log10(espectro + 1e-12), cmap='jet')
    plt.xlabel("tiempo (s)")
    plt.ylabel("frecuencia (Hz)")
    plt.title(f"Espectrograma - Clasificado como: {clasificacion}") # Título dinámico
    plt.colorbar(label="dB")
    plt.show()
    
def analizar_audio(senal, fm):
    """
    Procesa la señal de audio (recorte, espectrograma, extracción de 10 frecuencias dominantes).
    Ya NO grafica, solo extrae las características.
    """
    duracion_deseada = 20  
    num_muestras = duracion_deseada * fm
    
    if len(senal) > num_muestras:
        senal = senal[:num_muestras]
    
    # espectro (necesario para la extracción)
    frec, tiempo, espectro = spectrogram(
        senal,
        fs=fm,
        window=windows.hamming(512),
        nperseg=512,
        noverlap=256,
        nfft=1024,
        mode="magnitude"
    )

    # El código de graficación debe estar ELIMINADO o COMENTADO aquí.

    columnas_totales = espectro.shape[1]
    divisiones = np.linspace(0, columnas_totales, 11, dtype=int)
    frecuencias_dominantes = []

    for i in range(10):
        bloque = espectro[:, divisiones[i]:divisiones[i+1]]
        if bloque.size == 0:
            frecuencias_dominantes.append(0)
        else:
            energia = np.sum(bloque**2, axis=1)
            indice_max = np.argmax(energia)
            frecuencias_dominantes.append(frec[indice_max])

    return np.array(frecuencias_dominantes)

def cargar_base(ruta):
    """Carga los vectores de características de la base de datos."""
    archivos = sorted(os.listdir(ruta))
    datos = []
    for a in archivos:
        if a.endswith(".csv"):
            vector = pd.read_csv(os.path.join(ruta, a), header=None).values.flatten()
            datos.append(vector)
    return np.array(datos), len(archivos)

def validar_balanceo(mostrar_error=True):
    """Verifica si ambas bases de datos tienen el mismo número de vectores."""
    _, n1 = cargar_base(BASE_EPOC)
    _, n2 = cargar_base(BASE_SANOS)
    
    if n1 == 0 or n2 == 0:
        if mostrar_error:
            messagebox.showwarning("Advertencia", "Error: No puede haber carpetas de entrenamiento vacías.")
        return False

    if n1 != n2:
        if mostrar_error:
            messagebox.showwarning("Advertencia", f"Error: Las carpetas de entrenamiento no tienen el mismo número de vectores (EPOC: {n1}, SANOS: {n2}).")
        return False
        
    return True

def entrenar_bayes():
    """Entrena y devuelve el modelo Gaussian Naive Bayes."""
    base_epoc, _ = cargar_base(BASE_EPOC)
    base_sanos, _ = cargar_base(BASE_SANOS)
    X = np.vstack([base_epoc, base_sanos])
    Y = np.array([1]*len(base_epoc) + [0]*len(base_sanos))
    modelo = GaussianNB().fit(X, Y)
    return modelo

def realizar_diagnostico(archivo_wav, folio=None):
    """
    Realiza el diagnóstico de un archivo WAV dado, GRAFICA el resultado
    y lo guarda en la carpeta del paciente (si se proporciona folio).
    """
    if not validar_balanceo(mostrar_error=True):
        return None, None, "Base de datos no balanceada"

    try:
        # 1. Cargar y normalizar la señal (Necesitamos senal y fm para graficar)
        fm, senal_original = read(archivo_wav)
        senal = senal_original.astype(float)
        if senal.ndim == 2:
            senal = senal[:, 0]
        senal /= np.max(np.abs(senal))

        # 2. Analizar y Predecir
        frecuencias = analizar_audio(senal, fm)
        modelo = entrenar_bayes()

        prob_epoc = modelo.predict_proba([frecuencias])[0][1]
        pred = modelo.predict([frecuencias])[0]
        clasificacion = 'EPOC' if pred == 1 else 'SANO'
        
        # 3. Mostrar el espectrograma con la clasificación en el título
        mostrar_espectrograma(senal, fm, clasificacion) # ¡Añadido!

        # 4. Guardar registro si hay folio
        if folio is not None:
            carpeta_paciente = os.path.join("pacientes", f"paciente_{folio}", "diagnosticos")
            os.makedirs(carpeta_paciente, exist_ok=True)
            nombre_csv = os.path.basename(archivo_wav).replace(".wav", ".csv")
            pd.DataFrame([frecuencias]).to_csv(os.path.join(carpeta_paciente, nombre_csv), index=False, header=False)

        return clasificacion, prob_epoc, "Éxito"

    except Exception as e:
        return None, None, f"Error en el procesamiento: {e}"

# ==========================================================
# 3. FUNCIONES DE GESTIÓN DE AUDIO (Adaptadas para la nueva ventana)
# ==========================================================

def opcion_agregar(tipo):
    """Opción 1 y 2: Agregar audios a la base de datos de entrenamiento."""
    
    if tipo == 1:
        archivos = seleccionar_archivos("Seleccionar audios EPOC a agregar")
        carpeta = BASE_EPOC
        tipo_label = "EPOC"
    else: # tipo == 2
        archivos = seleccionar_archivos("Seleccionar audios SANOS a agregar")
        carpeta = BASE_SANOS
        tipo_label = "SANOS"
    
    if not archivos:
        messagebox.showinfo("Aviso", "No se seleccionó ningún archivo.")
        return
        
    for archivo in archivos:
        try:
            fm, senal = read(archivo)
            senal = senal.astype(float)
            if senal.ndim == 2:
                senal = senal[:, 0]
            senal /= np.max(np.abs(senal))

            frecuencias = analizar_audio(senal, fm)

            # Guardar en la base de datos de entrenamiento
            nombre = f"audio_{len(os.listdir(carpeta))+1}.csv"
            pd.DataFrame([frecuencias]).to_csv(os.path.join(carpeta, nombre), index=False, header=False)
        except Exception as e:
             messagebox.showerror("Error de Procesamiento", f"Error al procesar {os.path.basename(archivo)}: {e}")
             continue

    messagebox.showinfo("Éxito", f"{len(archivos)} audios ({tipo_label}) agregados correctamente a la base de entrenamiento.")


def opcion_agregar_y_diagnosticar():
    """Opción 3: Clasificar un audio y añadirlo a la base según el resultado."""
    
    if not validar_balanceo(mostrar_error=True):
        return

    archivos = seleccionar_archivos("Seleccionar audio(s) para clasificar y agregar")
    if not archivos:
        messagebox.showinfo("Aviso", "No se seleccionó ningún archivo.")
        return

    for archivo in archivos:
        clasificacion, prob_epoc, mensaje = realizar_diagnostico(archivo)
        
        if clasificacion is None:
            messagebox.showerror("Error", f"Fallo la clasificación de {os.path.basename(archivo)}: {mensaje}")
            continue

        # Convertir a vector de características (necesario para guardar en base)
        try:
            fm, senal = read(archivo)
            senal = senal.astype(float)
            if senal.ndim == 2:
                senal = senal[:, 0]
            senal /= np.max(np.abs(senal))
            frecuencias = analizar_audio(senal, fm)
        except Exception as e:
            messagebox.showerror("Error de Lectura", f"No se pudo analizar el audio {os.path.basename(archivo)}: {e}")
            continue
            
        
        # Guardar en la base de entrenamiento según la predicción
        if clasificacion == 'EPOC':
            carpeta = BASE_EPOC
        else:
            carpeta = BASE_SANOS

        nombre = f"audio_{len(os.listdir(carpeta))+1}.csv"
        pd.DataFrame([frecuencias]).to_csv(os.path.join(carpeta, nombre), index=False, header=False)
        
        messagebox.showinfo(
            "Resultado y Guardado",
            f"Archivo: {os.path.basename(archivo)}\n"
            f"Clasificación: **{clasificacion}**\n"
            f"Probabilidad de EPOC: {prob_epoc:.4f}\n"
            f"Guardado en base de entrenamiento: {os.path.basename(carpeta)}/{nombre}"
        )
        messagebox.showwarning("Advertencia", "Este proceso puede introducir sesgo en el modelo. Use con precaución.")


def opcion_clasificar_audio():
    """Opción 4: Clasificar un audio sin modificar la base de datos."""

    if not validar_balanceo(mostrar_error=True):
        return

    archivos = seleccionar_archivos("Seleccionar audio(s) para clasificar")
    if not archivos:
        messagebox.showinfo("Aviso", "No se seleccionó ningún archivo.")
        return

    resultados = []
    for archivo in archivos:
        clasificacion, prob_epoc, mensaje = realizar_diagnostico(archivo)
        
        if clasificacion is None:
            resultados.append(f"Archivo: {os.path.basename(archivo)}\nERROR: {mensaje}")
        else:
            resultados.append(
                f"Archivo: {os.path.basename(archivo)}\n"
                f"Clasificación: **{clasificacion}**\n"
                f"Probabilidad de EPOC: {prob_epoc:.4f}"
            )

    messagebox.showinfo("Resultados de Clasificación", "\n\n---\n\n".join(resultados))


def abrir_opciones_audio():
    """Crea la ventana con las opciones de gestión y clasificación de audios."""
    opciones = tk.Toplevel()
    opciones.title("Opciones de Gestión de Audio")
    opciones.configure(bg="#F5F7FA")
    opciones.resizable(False, False)

    frame = tk.Frame(opciones, bg="#F5F7FA", padx=20, pady=20)
    frame.pack(expand=True, fill="both")

    tk.Label(frame, text="Gestión de Audios y Clasificación", 
             font=("Open Sans", 24, "bold"), bg="#F5F7FA", fg="#001a33").pack(pady=15)
    
    # 1) Agregar EPOC
    tk.Button(frame, text="1) Agregar Audios EPOC a la Base de Entrenamiento",
              font=("Arial", 14), bg="#001F3F", fg="white", 
              command=lambda: opcion_agregar(1)).pack(fill="x", pady=5, ipadx=10, ipady=10)

    # 2) Agregar SANOS
    tk.Button(frame, text="2) Agregar Audios SANOS a la Base de Entrenamiento",
              font=("Arial", 14), bg="#001F3F", fg="white", 
              command=lambda: opcion_agregar(2)).pack(fill="x", pady=5, ipadx=10, ipady=10)

    # 3) Clasificar y Agregar
    tk.Button(frame, text="3) Clasificar Audio y Añadir a la Base (ADVERTENCIA: Puede Sesgar)",
              font=("Arial", 14), bg="#b72a27", fg="white", 
              command=opcion_agregar_y_diagnosticar).pack(fill="x", pady=15, ipadx=10, ipady=10)
              
    # 4) Clasificar sin Guardar
    tk.Button(frame, text="4) Clasificar Audio sin Guardar en la Base",
              font=("Arial", 14), bg="#001F3F", fg="white", 
              command=opcion_clasificar_audio).pack(fill="x", pady=5, ipadx=10, ipady=10)

    tk.Button(frame, text="Cerrar", command=opciones.destroy,
              font=("Arial", 12), bg="#CCCCCC", fg="#333").pack(pady=20)


# El resto del código de gestión de pacientes (toggle_grabacion, generar_folio, etc.) permanece igual.
# ... (funciones toggle_grabacion, generar_folio, abrir_carpeta_paciente, abrir_registro, abrir_detalle_paciente, abrir_historial)

def toggle_grabacion(folio, boton, ventana_detalle):
    """Inicia/detiene la grabación del audio y, al detener, llama al diagnóstico."""
    global grabando, stream, audio_buffer, fs_grabacion

    carpeta = os.path.join("pacientes", f"paciente_{folio}", "auscultaciones")
    os.makedirs(carpeta, exist_ok=True)

    if not grabando:
        # --- INICIAR GRABACIÓN ---
        grabando = True
        audio_buffer = []
        boton.config(text="Detener grabación", fg="red")
        messagebox.showinfo("Grabación", "Grabación iniciada. Presione 'Detener' para finalizar.")
   
        def callback(indata, frames, tiempo, status):
            if grabando:
                audio_buffer.append(indata.copy())

        stream = sd.InputStream(samplerate=fs_grabacion, channels=1, callback=callback)
        stream.start()

    else:
        # --- DETENER Y ANALIZAR GRABACIÓN ---
        grabando = False
        boton.config(text="Iniciar grabación", fg="#001a33")

        if stream:
            stream.stop()
            stream.close()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo = os.path.join(carpeta, f"auscultacion_{timestamp}.wav")
        audio_buffer_copia = audio_buffer.copy()
        
        if audio_buffer_copia:
            audio = np.concatenate(audio_buffer_copia, axis=0)
            write(archivo, fs_grabacion, audio)
            messagebox.showinfo("Guardado", f"Auscultación guardada en:\n{archivo}")
            
            # *** PUNTO CLAVE DE INTEGRACIÓN ***
            clasificacion, prob_epoc, mensaje = realizar_diagnostico(archivo, folio)

            if clasificacion is None:
                 messagebox.showerror("Error de Diagnóstico", f"No se pudo realizar el diagnóstico: {mensaje}")
            else:
                messagebox.showinfo(
                    "Resultado del Diagnóstico",
                    f"El audio grabado fue clasificado como: **{clasificacion}**\n"
                    f"Probabilidad de EPOC: {prob_epoc:.4f}"
                )
            
            # Volver a cargar el detalle del paciente para reflejar el nuevo diagnóstico/archivo
            ventana_detalle.destroy()
            abrir_detalle_paciente(folio) 
            
        else:
            messagebox.showwarning("Aviso", "No se grabó audio.")

def generar_folio():
    """Genera un nuevo número de folio para el paciente."""
    ruta_pacientes = "pacientes"
    if not os.path.exists(ruta_pacientes):
        os.makedirs(ruta_pacientes)
        return 1

    folios = []
    for carpeta in os.listdir(ruta_pacientes):
        if carpeta.startswith("paciente_"):
            try:
                num = int(carpeta.split("_")[1])
                folios.append(num)
            except:
                pass

    return max(folios) + 1 if folios else 1

def abrir_carpeta_paciente(folio):
    """Abre la carpeta del paciente en el explorador de archivos."""
    ruta = os.path.join("pacientes", f"paciente_{folio}")
    if not os.path.exists(ruta):
        messagebox.showerror("Error", f"No existe la carpeta: {ruta}")
        return
    try:
        os.startfile(ruta) # Esto funciona en Windows. En Linux/Mac podría requerir 'xdg-open' o 'open'
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo abrir la carpeta:\n{e}")

def abrir_registro():
    """Ventana para registrar un nuevo paciente."""
    reg = tk.Toplevel()
    reg.title("Registro de Paciente")
    reg.state("zoomed")
    reg.configure(bg="#ededed")

    # ... (el resto del código de la función abrir_registro es el mismo)
    edad_var = tk.StringVar()
    genero_var = tk.StringVar()
    fuma_var = tk.StringVar()
    padece_var = tk.StringVar()
    padecimiento_var = tk.StringVar(value="No presenta")

    card = tk.Frame(reg, bg="white", bd=5, relief="ridge")
    card.place(relx=0.5, rely=0.5, anchor="center", width=500, height=600)

    tk.Label(card, text="Registro de Paciente", font=("Open Sans", 32, "bold"),
             bg="white", fg="#b72a27").pack(pady=20)

    tk.Label(card, text="Edad:", bg="white", font=("Open Sans", 16),
             fg="#001a33").pack()
    tk.Entry(card, textvariable=edad_var, font=("Open Sans", 12),
             justify="center").pack()

    tk.Label(card, text="Género:", bg="white", font=("Open Sans", 16),
             fg="#001a33").pack(pady=(10, 0))
    ttk.Combobox(card, textvariable=genero_var, state="readonly",
                 values=["Hombre", "Mujer", "Otro"], font=("Open Sans", 12)).pack()

    tk.Label(card, text="¿Fuma?", bg="white", font=("Open Sans", 16),
             fg="#001a33").pack(pady=(10, 0))
    ttk.Combobox(card, textvariable=fuma_var, state="readonly",
                 values=["Sí", "No"], font=("Open Sans", 12)).pack()

    tk.Label(card, text="¿Sufre algún padecimiento?", bg="white",
             font=("Open Sans", 16), fg="#001a33").pack(pady=10)
    padece_combo = ttk.Combobox(card, textvariable=padece_var, state="readonly",
                                 values=["Sí", "No"], font=("Open Sans", 12))
    padece_combo.pack()

    padecimiento_label = tk.Label(card, text="¿Cuál?", bg="white",
                                     font=("Open Sans", 12))
    padecimiento_entry = tk.Entry(card, textvariable=padecimiento_var,
                                     font=("Open Sans", 12), justify="center")

    def mostrar_padecimiento(event):
        if padece_var.get() == "Sí":
            padecimiento_label.pack(pady=5)
            padecimiento_entry.pack()
            padecimiento_var.set("")
        else:
            padecimiento_label.pack_forget()
            padecimiento_entry.pack_forget()
            padecimiento_var.set("No presenta")

    padece_combo.bind("<<ComboboxSelected>>", mostrar_padecimiento)

    def guardar_datos():
        edad = edad_var.get()
        genero = genero_var.get()
        fuma = fuma_var.get()
        padecimiento = padecimiento_var.get()

        if not edad or not genero or not fuma:
            messagebox.showerror("Error", "Debe llenar todos los campos obligatorios.")
            return

        folio = generar_folio()

        ruta_carpeta = os.path.join("pacientes", f"paciente_{folio}")
        os.makedirs(ruta_carpeta, exist_ok=True)

        ruta_csv = os.path.join(ruta_carpeta, "datos.csv")

        with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Folio", "Edad", "Género", "Fuma", "Padecimiento"])
            writer.writerow([folio, edad, genero, fuma, padecimiento])

        messagebox.showinfo("Registro", f"Paciente guardado con folio #{folio}")

        edad_var.set("")
        genero_var.set("")
        fuma_var.set("")
        padece_var.set("")
        padecimiento_var.set("No presenta")

    tk.Button(card, text="Guardar", command=guardar_datos,
              bg="#001F3F", fg="white", font=("Arial", 14)).pack(side="bottom", pady=20)

    tk.Button(card, text="Salir", command=reg.destroy,
              bg="#001F3F", fg="white", font=("Arial", 14)).pack(side="bottom", pady=10)


def abrir_detalle_paciente(folio):
    """
    Función que crea la ventana de detalle del paciente (antes 'abrir_detalle' dentro de 'abrir_historial').
    Se separa para poder recargarla después del diagnóstico.
    """
    carpeta_paciente = os.path.join("pacientes", f"paciente_{folio}")
    archivo_csv = os.path.join(carpeta_paciente, "datos.csv")   

    if not os.path.exists(archivo_csv):
        messagebox.showerror("Error", f"No se encontró datos.csv del Paciente {folio}")
        return
    
    detalle = tk.Toplevel()             
    detalle.title(f"Detalle Paciente #{folio}")
    detalle.state("zoomed")
    detalle.configure(bg="#ededed")

    # Cargar imágenes de la GUI (asegúrate de que estas rutas sean correctas en tu entorno)
    try:
        # Nota: Usamos las mismas imágenes genéricas que antes para simplificar la adaptación
        img_ta = Image.open("Usuario.jpeg").resize((90, 90))
        detalle.img_ta = ImageTk.PhotoImage(img_ta)

        img_ea = Image.open("Folio.jpeg").resize((90, 90))
        detalle.img_ea = ImageTk.PhotoImage(img_ea)

        img_as = Image.open("Usuario.jpeg").resize((350, 350))
        detalle.img_as = ImageTk.PhotoImage(img_as)
    except FileNotFoundError as e:
        messagebox.showerror("Error de Imagen", f"No se encontró un archivo de imagen. Verifica la ruta: {e}")
        return

    # --- SIDEBAR (GRABACIÓN) ---
    sidebar = tk.Frame(detalle, bg="white", width=450)
    sidebar.pack(side="left", fill="y")
    sidebar.pack_propagate(False)

    tk.Label(sidebar, image=detalle.img_as, bg="white").pack(pady=20)
    
    btn_toggle = tk.Button(sidebar,text="Iniciar grabación",image=detalle.img_ta,
        compound="left",font=("Open Sans", 20, "bold"),fg="#001a33",bg="white",relief="flat")
    # El comando llama a toggle_grabacion, pasando el folio, el botón y la ventana de detalle para recargar
    btn_toggle.config(command=lambda f=folio, b=btn_toggle, d=detalle: toggle_grabacion(f, b, d))
    btn_toggle.pack(fill="x", pady=10)

    tk.Button(sidebar, text="Abrir carpeta", image=detalle.img_ea, compound="left", font=("Open Sans", 20, "bold"),
                  fg="#001a33", bg="white", relief="flat", command=lambda f=folio: abrir_carpeta_paciente(f),padx= 70).pack(fill="x", pady=10)

    # --- CONTENEDOR PRINCIPAL (DATOS) ---
    contenedor = tk.Frame(detalle, bg="#ededed")
    contenedor.pack(side="left", fill="both", expand=True)

    with open(archivo_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        fila = next(reader)

    tk.Label(contenedor, text=f"Paciente #{folio}",font=("Open Sans", 40, "bold"),
             bg="#ededed", fg="#001a33").pack(pady=20)

    labels = ["Folio", "Edad", "Género", "Fuma", "Padecimiento"]

    for i, label in enumerate(labels):
        tk.Label(contenedor, text=f"{label}: {fila[i]}",
                 font=("Arial", 18), bg="#ededed", fg="#333").pack(anchor="w", pady=8)

    # Mostrar Diagnósticos Anteriores (Nuevo)
    tk.Label(contenedor, text="Último Diagnóstico:",
             font=("Arial", 18, "bold"), bg="#ededed", fg="#b72a27").pack(anchor="w", pady=(20, 5))
    
    # Intenta leer el último diagnóstico guardado (si existe)
    diagnosticos_path = os.path.join(carpeta_paciente, "diagnosticos")
    ultimo_diagnostico = "Pendiente o sin datos."
    
    if os.path.exists(diagnosticos_path):
        archivos = sorted([f for f in os.listdir(diagnosticos_path) if f.endswith(".csv")], reverse=True)
        if archivos:
            try:
                # Simplemente verifica el nombre para el timestamp
                timestamp = archivos[0].replace("auscultacion_", "").replace(".csv", "")
                ultimo_diagnostico = f"Última Auscultación: {timestamp}"
            except:
                 pass

    tk.Label(contenedor, text=ultimo_diagnostico,
             font=("Arial", 16), bg="#ededed", fg="#333").pack(anchor="w", pady=5)


    tk.Frame(contenedor, bg="#ededed").pack(expand=True) 
    tk.Button(contenedor, text="Cerrar",
              bg="#001F3F", fg="white", font=("Arial", 16),
              command=detalle.destroy).pack(pady=40)


def abrir_historial():
    """Ventana que muestra la tabla de historial de pacientes."""

    def abrir_detalle(event):
        seleccion = tabla.focus()
        if not seleccion:
            return

        datos = tabla.item(seleccion, "values")
        folio = datos[0]
        # Llama a la función de detalle separada
        abrir_detalle_paciente(folio)

    historial = tk.Toplevel()
    historial.title("Historial de Pacientes")
    historial.state("zoomed")

    # ... (el resto del código de la función abrir_historial es el mismo)
    frame = tk.Frame(historial, bg="#F5F7FA")
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text="Historial de Pacientes",
             font=("Arial", 36, "bold"), bg="#F5F7FA", fg="#001a33").pack(pady=20)

    tabla_frame = tk.Frame(frame, bg="white", bd=2, relief="groove")
    tabla_frame.pack(pady=20, padx=40, fill="both", expand=True)

    columnas = ("Folio", "Edad", "Género", "Fuma", "Padecimiento")
    tabla = ttk.Treeview(tabla_frame, columns=columnas, show="headings")
    tabla.pack(fill="both", expand=True)

    for col in columnas:
        tabla.heading(col, text=col)
        tabla.column(col, width=200)

    tabla.bind("<Double-1>", abrir_detalle)

    ruta_pacientes = "pacientes"
    registros = False

    if os.path.exists(ruta_pacientes):
        for carpeta in os.listdir(ruta_pacientes):

            ruta_carpeta = os.path.join(ruta_pacientes, carpeta)

            if os.path.isdir(ruta_carpeta) and carpeta.startswith("paciente_"):

                archivo_csv = os.path.join(ruta_carpeta, "datos.csv")

                if os.path.exists(archivo_csv):

                    with open(archivo_csv, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        next(reader, None)
                        fila = next(reader, None)

                        if fila:
                            tabla.insert("", tk.END, values=fila)
                            registros = True

    if not registros:
        messagebox.showinfo("Historial", "Aún no hay registros guardados.")

    tk.Button(frame, text="Salir", command=historial.destroy,
              bg="#001F3F", fg="white", font=("Arial", 14)).pack(side="bottom", pady=20)


class Principal(tk.Tk):
    """Clase principal de la aplicación GUI."""
    def __init__(self):
        super().__init__()

        self.title("JYDH-EPOC")
        self.state("zoomed")
        self.configure(bg="#001a33")

        # Cargar imágenes (Verifica las rutas)
        try:
            # Aquí se usan las imágenes para los botones del menú.
            img_r = Image.open("Usuario.jpeg").resize((90, 90))
            self.img_r = ImageTk.PhotoImage(img_r)

            img_h = Image.open("Folio.jpeg").resize((90, 90))
            self.img_h = ImageTk.PhotoImage(img_h)
            
            # Imagen genérica para el nuevo botón (puedes cambiarla)
            img_a = Image.open("Logo.jpeg").resize((90, 90)) 
            self.img_a = ImageTk.PhotoImage(img_a)

            img_logo = Image.open("Logo.jpeg").resize((350, 350))
            self.icono = ImageTk.PhotoImage(img_logo)
        except FileNotFoundError as e:
            messagebox.showerror("Error de Imagen", f"No se encontró un archivo de imagen. Verifica la ruta: {e}")
            self.destroy()
            return

        # --- SIDEBAR ---
        sidebar = tk.Frame(self, bg="white", width=350)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        logo = tk.Label(sidebar, text="Menú", font=("Open Sans", 44, "bold"),
                         fg="#b72a27", bg="white")
        logo.pack(pady=30)

        # Botón 1: Registrar Paciente
        tk.Button(sidebar, text="Registrar\npaciente", image=self.img_r,
                  compound="left", font=("Open Sans", 30, "bold"),
                  fg="#001a33", bg="white", relief="flat", command=abrir_registro).pack(fill="x", pady=30)

        # Botón 2: Historial
        tk.Button(sidebar, text="Historial", image=self.img_h,
                  compound="left", font=("Open Sans", 30, "bold"),
                  fg="#001a33", bg="white", relief="flat", command=abrir_historial).pack(fill="x", pady=30)

        # *** NUEVO BOTÓN: Seleccionar Audios ***
        tk.Button(sidebar, text="Seleccionar\nAudios", image=self.img_a,
                  compound="left", font=("Open Sans", 30, "bold"),
                  fg="#001a33", bg="white", relief="flat", command=abrir_opciones_audio).pack(fill="x", pady=30)
        
        # Botón 3: Salir
        tk.Button(sidebar, text="Salir", font=("Open Sans", 30, "bold"),
                  fg="#001a33", bg="white", relief="flat", command=self.destroy).pack(fill="x", pady=30)

        # --- MAIN CONTENT ---
        main = tk.Frame(self, bg="#ededed")
        main.pack(side="right", expand=True, fill="both")

        tk.Label(main, image=self.icono, bg="#ededed").pack(pady=50)

        tk.Label(main, text="Bienvenido a", font=("Open Sans", 32, "bold"),
                 fg="#001a33", bg="#ededed").pack()

        tk.Label(main, text="JYDHI", font=("Open Sans", 48, "bold"),
                 fg="#b72a27", bg="#ededed").pack()

if __name__ == "__main__":
    app = Principal()
    app.mainloop()