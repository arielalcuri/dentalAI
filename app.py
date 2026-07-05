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

    st.write("**Instrucciones:**")
    st.write("1. Sube una radiografía panorámica en el área central.")
    st.write("2. La IA la procesará automáticamente.")
    st.write("3. Revisa el resumen de hallazgos en la parte inferior.")

# Título principal
st.title("Diagnóstico de Radiografías Panorámicas")
st.write("Sube una imagen y nuestra Inteligencia Artificial detectará y remarcará posibles caries y lesiones.")

with st.expander("⚙️ Opciones Avanzadas"):
    conf_threshold = st.slider(
        "Nivel de Confianza de la IA", 
        min_value=0.05, max_value=0.95, value=0.25, step=0.05, 
        help="Aumenta este valor si ves ruido o duplicados. Bájalo si la IA no detecta algo evidente."
    )
    st.info("💡 Recomendación: Mantener en 0.25 para diagnósticos estándar. Bajar la sensibilidad puede revelar patologías sutiles, pero aumentará los falsos positivos.")
st.write("---")
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
        # Reemplazamos st.image por Plotly para que ambas columnas tengan idéntica altura y capacidades de zoom
        fig_left = go.Figure()
        fig_left.add_layout_image(
            dict(
                source=image, xref="x", yref="y", x=0, y=0,
                sizex=image.width, sizey=image.height, sizing="stretch", opacity=1, layer="below"
            )
        )
        fig_left.update_xaxes(showgrid=False, range=(0, image.width), showticklabels=False)
        fig_left.update_yaxes(showgrid=False, scaleanchor="x", range=(image.height, 0), showticklabels=False)
        fig_left.update_layout(
            height=450,
            margin=dict(l=0, r=0, b=0, t=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode=False
        )
        st.plotly_chart(fig_left, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})
    
    # Animación de carga mientras la IA piensa
    with col2:
        st.subheader("🖼️ Diagnóstico de la IA")
        with st.spinner("La Inteligencia Artificial está analizando la imagen y detectando piezas dentales..."):
            # Realizar predicción con la sensibilidad elegida por el usuario
            results = model.predict(source=image, conf=conf_threshold)
            
            # El resultado viene en una lista. Tomamos el primero.
            result = results[0]
            
            # Llamar a Roboflow para detectar piezas dentales
            roboflow_preds = []
            try:
                import base64
                from io import BytesIO
                buffered = BytesIO()
                # Convertimos a RGB en caso de que sea RGBA
                if image.mode != 'RGB':
                    rgb_image = image.convert('RGB')
                else:
                    rgb_image = image
                rgb_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                api_key = st.secrets.get("ROBOFLOW_API_KEY", "8CPDdW5h7dJTljlpbiNC")
                url = "https://detect.roboflow.com/teeth-detection-and-numbering-agi2i/18"
                resp = requests.post(
                    url, 
                    params={"api_key": api_key}, 
                    data=img_str, 
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                if resp.status_code == 200:
                    roboflow_preds = resp.json().get("predictions", [])
            except Exception as e:
                print("Error Roboflow:", e)
            
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
        fig.update_xaxes(showgrid=False, range=(0, image.width), showticklabels=False)
        fig.update_yaxes(showgrid=False, scaleanchor="x", range=(image.height, 0), showticklabels=False)
        fig.update_layout(
            height=450,
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
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})
        
    st.write("---")
    st.subheader("📋 Resumen Clínico de Hallazgos")
    
    # Extraer las detecciones
    boxes = result.boxes
    if len(boxes) == 0:
        st.success("✅ **¡Buenas noticias!** No se detectaron anomalías visibles con el nivel de sensibilidad actual.")
    else:
        st.warning(f"⚠️ La Inteligencia Artificial detectó **{len(boxes)}** posible(s) hallazgo(s) anatómico(s) o patológico(s).")
        
        # Preparar los datos para una tabla bonita cruzando patologías con piezas dentales
        hallazgos = []
        for box in boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            
            # Coordenadas de la patología local
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Buscar el diente de roboflow que contenga este centroide
            pieza = "General"
            for p in roboflow_preds:
                px = p["x"]
                py = p["y"]
                pw = p["width"]
                ph = p["height"]
                # Bounding box del diente detectado por roboflow
                rx1 = px - pw/2
                rx2 = px + pw/2
                ry1 = py - ph/2
                ry2 = py + ph/2
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    pieza = p["class"]
                    break
            
            hallazgos.append({"pieza": pieza, "class_name": class_name})

        # Agrupar por nombre y pieza
        from collections import Counter
        conteo = Counter((h["class_name"], h["pieza"]) for h in hallazgos)
            
        import pandas as pd
        datos_tabla = []
        for (name, pieza), count in conteo.items():
            descripcion = iccms_desc.get(name, "")
            datos_tabla.append({
                "Cantidad": count,
                "Pieza Dental": pieza,
                "Identificador AI (Inglés)": name,
                "Diagnóstico y Descripción (Español)": descripcion
            })
            
        # Mostrar tabla interactiva
        df_resultados = pd.DataFrame(datos_tabla)
        st.dataframe(df_resultados, use_container_width=True, hide_index=True)

st.markdown("<br><hr><center><i>Desarrollado estrictamente como herramienta de diagnóstico presuntivo asistido. El diagnóstico final debe ser emitido por un profesional odontólogo.</i></center>", unsafe_allow_html=True)
