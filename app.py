import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
import requests
from io import BytesIO
import plotly.graph_objects as go

# Configuración de la página para que se vea más ancha y profesional
st.set_page_config(
    page_title="IA Odontológica",
    page_icon="🦷",
    layout="wide"
)

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

# Panel Lateral de Información
with st.sidebar:
    st.title("🦷 IA Odontológica")
    st.write("Bienvenido al sistema de diagnóstico asistido por computadora.")
    st.info(
        "Este modelo ha sido entrenado para detectar "
        "y segmentar patologías dentales en radiografías panorámicas."
    )
    st.write("---")
    conf_threshold = st.slider(
        "Sensibilidad de la IA", 
        min_value=0.05, max_value=0.95, value=0.25, step=0.05, 
        help="Aumenta este valor si ves predicciones duplicadas o ruido. Bájalo si la IA no detecta algo evidente."
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
    # Usar columnas para ponerlas lado a lado (Streamlit las apila en celulares automáticamente)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Imagen Original")
        st.image(image, use_container_width=True)
    
    # Animación de carga mientras la IA piensa
    with col2:
        st.subheader("🖼️ Diagnóstico de la IA")
        with st.spinner("La Inteligencia Artificial está analizando la imagen..."):
            # Realizar predicción con la sensibilidad elegida por el usuario
            results = model.predict(source=image, conf=conf_threshold)
            
            # El resultado viene en una lista. Tomamos el primero.
            result = results[0]
            
        # Crear gráfico interactivo con Plotly
        fig = go.Figure()
        
        # Añadir la imagen original de fondo
        fig.add_layout_image(
            dict(
                source=image,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=image.width,
                sizey=image.height,
                sizing="stretch",
                opacity=1,
                layer="below"
            )
        )
        fig.update_xaxes(showgrid=False, range=(0, image.width), showticklabels=False, fixedrange=True)
        fig.update_yaxes(showgrid=False, scaleanchor="x", range=(image.height, 0), showticklabels=False, fixedrange=True)
        fig.update_layout(
            height=image.height,
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="closest"
        )
        
        boxes = result.boxes
        masks = result.masks
        
        if masks is not None:
            added_legends = set()
            for idx, mask in enumerate(masks.xy):
                class_id = int(boxes.cls[idx].item())
                conf = float(boxes.conf[idx].item())
                class_name = model.names[class_id]
                
                if len(mask) == 0:
                    continue
                    
                x_coords = mask[:, 0].tolist()
                y_coords = mask[:, 1].tolist()
                
                # Cerrar el polígono
                x_coords.append(x_coords[0])
                y_coords.append(y_coords[0])
                
                desc_espanol = iccms_desc.get(class_name, "")
                hover_text = f"<b>{class_name}</b><br><i>{desc_espanol}</i><br>Confianza: {conf:.0%}"
                
                show_legend = class_name not in added_legends
                added_legends.add(class_name)
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords, 
                    fill="toself",
                    mode="lines",
                    line=dict(width=2),
                    name=class_name,
                    text=hover_text,
                    hoverinfo="text",
                    opacity=0.4,
                    showlegend=show_legend,
                    legendgroup=class_name
                ))
        elif boxes is not None:
            added_legends = set()
            # Fallback por si el modelo detecta cajas pero no máscaras
            for idx, box in enumerate(boxes.xyxy):
                class_id = int(boxes.cls[idx].item())
                conf = float(boxes.conf[idx].item())
                class_name = model.names[class_id]
                
                x1, y1, x2, y2 = box.tolist()
                x_coords = [x1, x2, x2, x1, x1]
                y_coords = [y1, y1, y2, y2, y1]
                
                desc_espanol = iccms_desc.get(class_name, "")
                hover_text = f"<b>{class_name}</b><br><i>{desc_espanol}</i><br>Confianza: {conf:.0%}"
                
                show_legend = class_name not in added_legends
                added_legends.add(class_name)
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords, 
                    fill="toself",
                    mode="lines",
                    line=dict(width=2),
                    name=class_name,
                    text=hover_text,
                    hoverinfo="text",
                    opacity=0.4,
                    showlegend=show_legend,
                    legendgroup=class_name
                ))
        
        # Mostrar el gráfico interactivo
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})
        
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
            
        for name, count in class_counts.items():
            descripcion = iccms_desc.get(name, "")
            st.write(f"- **{count}** lesión(es) clasificada(s) como: `{name}` ➔ **{descripcion}**")

st.markdown("<br><hr><center><i>Desarrollado estrictamente como herramienta de diagnóstico presuntivo asistido. El diagnóstico final debe ser emitido por un profesional odontólogo.</i></center>", unsafe_allow_html=True)
