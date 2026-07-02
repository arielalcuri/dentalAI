import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
import requests
from io import BytesIO

# Configuración de la página para que se vea más ancha y profesional
st.set_page_config(
    page_title="IA Odontológica",
    page_icon="🦷",
    layout="wide"
)

# Panel Lateral de Información
with st.sidebar:
    st.title("🦷 IA Odontológica")
    st.write("Bienvenido al sistema de diagnóstico asistido por computadora.")
    st.info(
        "Este modelo ha sido entrenado para detectar "
        "y segmentar patologías dentales en radiografías panorámicas."
    )
    st.write("---")
    st.write("**Instrucciones:**")
    st.write("1. Sube una radiografía panorámica en el área central.")
    st.write("2. La IA la procesará automáticamente.")
    st.write("3. Revisa el resumen de hallazgos en la parte inferior.")

# Título principal
st.title("Diagnóstico de Radiografías Panorámicas")
st.write("Sube una imagen y nuestra Inteligencia Artificial detectará y remarcará posibles caries y lesiones.")

# Cargar el modelo (se cachea para que no recargue desde el disco en cada interacción)
@st.cache_resource
def load_model():
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        st.error(f"No se encontró el modelo entrenado en '{model_path}'. Asegúrate de haberlo copiado a esta carpeta.")
        st.stop()
    return YOLO(model_path)

model = load_model()

# Opciones de entrada de imagen
upload_option = st.radio("¿Cómo quieres cargar la radiografía?", ("Subir archivo desde mi PC", "Pegar enlace (URL) de internet"))

image = None

if upload_option == "Subir archivo desde mi PC":
    uploaded_file = st.file_uploader("Elige una radiografía (JPG, PNG)...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
else:
    image_url = st.text_input("Pega aquí el enlace directo a la imagen:")
    if image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error("No se pudo cargar la imagen desde ese enlace. Asegúrate de que sea un enlace directo que termine en .jpg o .png.")

if image is not None:
    # Mostrar la imagen original
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_container_width=True)
    
    # Animación de carga mientras la IA piensa
    with st.spinner("La Inteligencia Artificial está analizando la imagen..."):
        # Realizar predicción con una confianza baja para no perderse nada (0.15)
        results = model.predict(source=image, conf=0.15)
        
        # El resultado viene en una lista. Tomamos el primero.
        result = results[0]
        
        # Generar imagen con los dibujos de colores (Plotting)
        # result.plot() devuelve un array numpy en formato BGR (OpenCV)
        im_array = result.plot()
        # Invertimos los canales de BGR a RGB para que se vea bien en la web
        im_rgb = im_array[..., ::-1]
        result_image = Image.fromarray(im_rgb)
        
    with col2:
        st.subheader("Diagnóstico de la IA")
        st.image(result_image, use_container_width=True)
        
    st.write("---")
    st.subheader("📋 Resumen de Hallazgos")
    
    # Extraer las detecciones para mostrarlas en texto
    boxes = result.boxes
    if len(boxes) == 0:
        st.success("¡Buenas noticias! No se detectaron anomalías con el nivel de confianza actual.")
    else:
        st.warning(f"Se detectaron {len(boxes)} posible(s) problema(s).")
        
        # Contar cuántas veces aparece cada clase detectada
        class_counts = {}
        for box in boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        # Diccionario de traducción clínica (Modelo Hipercompleto)
        iccms_desc = {
            "RA1": "Caries inicial (Mancha blanca en esmalte)",
            "RA2": "Caries leve (Esmalte interno sin cavidad)",
            "RA3": "Caries moderada (Ruptura de esmalte localizada)",
            "RB4": "Caries profunda (Sombra subyacente en dentina)",
            "RC5": "Caries profunda (Tercio interno de dentina / cavitada)",
            "RC6": "Caries muy profunda (Pulpa expuesta / gran cavidad)",
            "Bone Loss": "Pérdida Ósea (Enfermedad Periodontal)",
            "Caries": "Caries (Genérica)",
            "Crown": "Corona Protésica",
            "Cyst": "Quiste Dental",
            "Filling": "Restauración / Empaste (Resina o Amalgama)",
            "Fracture teeth": "Fractura Dental",
            "Implant": "Implante Dental",
            "Malaligned": "Diente Malalineado",
            "Mandibular Canal": "Conducto Dentario Inferior (Nervio)",
            "Missing teeth": "Diente Ausente",
            "Periapical lesion": "Lesión Periapical (Absceso/Infección)",
            "Permanent Teeth": "Diente Permanente",
            "Primary teeth": "Diente Primario (De leche)",
            "Retained root": "Resto Radicular (Raíz retenida)",
            "Root Canal Treatment": "Tratamiento de Conducto (Endodoncia)",
            "Root Piece": "Fragmento Radicular",
            "Root resorption": "Reabsorción Radicular",
            "Supra Eruption": "Extrusión Dental (Sobre-erupcionado)",
            "TAD": "Microtornillo de Ortodoncia (TAD)",
            "abutment": "Pilar de Implante (Abutment)",
            "attrition": "Atrición (Desgaste dental)",
            "bone defect": "Defecto Óseo",
            "gingival former": "Cicatriceador Gingival (Implante)",
            "impacted tooth": "Diente Retenido / Impactado",
            "maxillary sinus": "Seno Maxilar",
            "metal band": "Banda Metálica (Ortodoncia)",
            "orthodontic brackets": "Brackets de Ortodoncia",
            "permanent retainer": "Retenedor Permanente Fijo",
            "plating": "Placa de Osteosíntesis",
            "post - core": "Perno Muñón",
            "wire": "Arco o Alambre de Ortodoncia"
        }
        
        for name, count in class_counts.items():
            descripcion = iccms_desc.get(name, "")
            st.write(f"- **{count}** lesión(es) clasificada(s) como: `{name}` ➔ **{descripcion}**")

st.markdown("<br><hr><center><i>Desarrollado estrictamente como herramienta de diagnóstico presuntivo asistido. El diagnóstico final debe ser emitido por un profesional odontólogo.</i></center>", unsafe_allow_html=True)
