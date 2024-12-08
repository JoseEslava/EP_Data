import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

st.set_page_config(page_title="EPData Modelos", page_icon="bar-chart")
st.header('Modelos de Inteligencia Artificial')

df_intel = pd.read_csv('anticorrupcion_intel.csv', sep=',', header=0)

map_entidad_federativa = np.array([
    'Aguascalientes',
    'Baja California',
    'Baja California Sur',
    'Campeche',
    'Coahuila de Zaragoza',
    'Colima',
    'Chiapas',
    'Chihuahua',
    'Ciudad de México',
    'Durango',
    'Guanajuato',
    'Guerrero',
    'Hidalgo',
    'Jalisco',
    'México',
    'Michoacán de Ocampo',
    'Morelos',
    'Nayarit',
    'Nuevo León',
    'Oaxaca',
    'Puebla',
    'Querétaro',
    'Quintana Roo',
    'San Luis Potosí',
    'Sinaloa',
    'Sonora',
    'Tabasco',
    'Tamaulipas',
    'Tlaxcala',
    'Veracruz de Ignacio de la Llave',
    'Yucatán',
    'Zacatecas'
])

map_organismo_poder = np.array([
    'Organismo público descentralizado',
    'Gobierno estatal',
    'Empresa Productiva del Estado',
    'Dependencia Centralizada Federal',
    'Organismo desconcentrado federal',
    'Organismo paraestatal',
    'Organismo Público Autónomo',
    'Fideicomiso Público Sectorizado',
    'Poder de gobierno ',
    'Gobierno municipal',
    'Asociación Cívil Sectorizada',
    'Órgano especializado del Poder Judicial',
    'Institución Nacional de Seguros y Fianzas',
])

map_funcion_inst = np.array([
    'Seguridad Social',
    'Gobierno',
    'Generación de energía',
    'Orden y Justicia',
    'Banca de desarrollo',
    'Salud',
    'Desarrollo de Estadística',
    'Regulación físcal y aduanera',
    'Educación',
    'Bienestar social',
    'Financiamiento y comercialización',
    'Regulación',
    'Comunicaciones y Transportes',
    'Desarrollo de población en vulnerabilidad',
    'Agricultura, ganadería, desarrollo rural y pesca',
    'Desarrollo territorial',
    'Investigación y Desarrollo',
    'Transparencia',
    'Fomento al turismo',
    'Proceso electoral y transparencia',
    'Cultura',
    'Trabajo',
    'Hacienda',
    'Medio Ambiente y Recursos Naturales',
    'Protección al Medio Ambiente y Recuersos Naturales',
    'Economía y comercio',
    'Gestión inmobiliaria',
    'Comercialización',
    'Banca nacional',
    'Servicios de agua',
    'Impartición de justicia',
    'Financiamiento hipotecario',
    'Obtención de recursos financieros',
    'Regulación y coinciliación',
    'Relaciones exteriores',
    'Producción de hidrocarburos',
    'Laboratorio',
    'Seguros',
    'Deporte',
    'Derechos Humanos'
])

map_autoridad_sancionadora = np.array([
    'ORGANO INTERNO DE CONTROL',
    'CONTRALORIA DE ESTADO',
    'JUZGADOS DE DISTRITO',
    'CONTRALORIA MUNICIPAL',
    'CONGRESO ESTATAL',
    'SFP (CONTRALORIA INTERNA)',
    'CONTRALORIA GENERAL DEL D.F.',
    'CONTRALORIA GENERAL',
    'DIRECCION GENERAL DE RESPONSABILIDADES Y SITUACION PATRIMONIAL (SFP)',
    'TRIBUNAL SUPERIOR DE JUSTICIA ESTATAL',
    'CONSEJO DE LA JUDICATURA FEDERAL',
    'SUPREMO TRIBUNAL MILITAR',
    'PODER JUDICIAL DEL ESTADO',
    'TRIBUNAL DE LO CONTENCIOSO ADMINISTRATIVO',
    'SECRETARIA DE LA FUNCION PUBLICA',
    'JUNTA FEDERAL DE CONCILIACION Y ARBITRAJE',
    'JUZGADO DEL FUERO COMUN',
    'TRIBUNALES UNITARIOS DE CIRCUITO',
    'CONTRALORIA INTERNA (TRIBUNALES)',
    'TRIBUNAL FEDERAL DE JUSTICIA ADMINISTRATIVA',
    'CONTRALORIA INTERNA',
    'CONTRALORIA INTERNA (PODER JUDICIAL)',
    'SUPREMA CORTE DE JUSTICIA DE LA NACION',
    'COMISION DE HONOR Y JUSTICIA'
])

map_causa_motivo_hechos = np.array([
    'NEGLIGENCIA ADMINISTRATIVA',
    'DELITO COMETIDO POR SERVIDORES PUBLICOS',
    'ABUSO DE AUTORIDAD',
    'INCUMPLIMIENTO EN DECLARACION DE SITUACION PATRIMONIAL',
    'COHECHO O EXTORSION',
    'VIOLACION LEYES Y NORMATIVIDAD PRESUPUESTAL',
    'VIOLACION PROCEDIMIENTOS DE CONTRATACION',
    'EJERCICIO INDEBIDO DE SUS FUNCIONES EN MATERIA MIGRATORIA',
    'VIOLACIÓN A LOS DERECHOS HUMANOS'
])

map_genero = np.array(['Hombre', 'Mujer'])
map_nivel_salarial = np.array([1,2,3,4,5,6,7,8,9,10])

# called back models created before
model1 =pickle.load(open("xgb_model","rb"))
model2= pickle.load(open("rf_model","rb"))

# selectbox method and append models to give a choice clients
models = st.sidebar.selectbox("Selecciona el modelo",("Random Forest | Abuso de autoridad","XGBoost | Abuso sexual") )

# Create a sidebar for user input
st.sidebar.title("Abuso de Autoridad")
st.sidebar.title("Abuso Sexual")
st.sidebar.markdown("Escribe aquí tus preferencias:")

entidad = st.sidebar.selectbox("Entidad federativa", map_entidad_federativa)
organismo = st.sidebar.selectbox("Organismo de poder", map_organismo_poder)
funcion = st.sidebar.selectbox("Función de la institución", map_funcion_inst)
nivelSalarial = st.sidebar.selectbox("Nivel salarial (1-Max 10-Min)", map_nivel_salarial)
genero = st.sidebar.selectbox("Género", map_genero)

if models == "Random Forest | Abuso de autoridad":
    autoridad = st.sidebar.selectbox("Autoridad Sancionadora", map_autoridad_sancionadora)
    
    entidad_ = np.where(map_entidad_federativa == entidad)[0][0]
    genero_ = np.where(map_genero == genero)[0][0]
    nivelSalarial_ = np.where(map_nivel_salarial == nivelSalarial)[0][0]
    autoridad_ = np.where(map_autoridad_sancionadora == autoridad)[0][0]
    organismo_ = np.where(map_organismo_poder == organismo)[0][0]
    funcion_ = np.where(map_funcion_inst == funcion)[0][0]

    caso_rfc = np.array([[funcion_+1, entidad_+1, nivelSalarial_+1, autoridad_+1, organismo_+1, genero_]])
    print(funcion_+1, entidad_+1, nivelSalarial_+1, autoridad_+1, organismo_+1, genero_)

    prediccion = model2.predict(caso_rfc)
    probabilidad_abuso_autoridad = model2.predict_proba(caso_rfc)[0, 1]  # Probabilidad de default (clase 1)

    # Ciclo para obtener la probabilidad de abuso de autoridad por estado
    probabilid_edos = []
    for i in range(1, 33):
        caso_estado = np.array([[funcion_+1, i, nivelSalarial_+1, autoridad_+1, organismo_+1, genero_]])
        probabilidad_abuso_estado = model2.predict_proba(caso_estado)[0, 1]

        # Almacenar resultado
        probabilid_edos.append({
            "CVE_ENT": i,
            "Metrica": probabilidad_abuso_estado
        })

    df_edos_abuso = pd.DataFrame(probabilid_edos)
    df_edos_abuso.insert(loc=1, column='entidad_federativa', value=map_entidad_federativa)

    if prediccion[0] == 1:
        st.error("El modelo Random Forest predice que el servidor público realiza algún tipo de Abuso de autoridad")
    else:
        st.warning("El modelo Random Forest predice que el servidor público podría realizar algún tipo de Abuso de autoridad")
    
    if probabilidad_abuso_autoridad > 0.5:
        st.error(f"La probabilidad de que un servidor público realice algún tipo de abuso de autoridad: {probabilidad_abuso_autoridad:.2f}")
    else:
        st.warning(f"La probabilidad de que un servidor público realice algún tipo de abuso de autoridad: {probabilidad_abuso_autoridad:.2f}")
else :
    entidad_ = np.where(map_entidad_federativa == entidad)[0][0]
    genero_ = np.where(map_genero == genero)[0][0]
    nivelSalarial_ = np.where(map_nivel_salarial == nivelSalarial)[0][0]
    organismo_ = np.where(map_organismo_poder == organismo)[0][0]
    funcion_ = np.where(map_funcion_inst == funcion)[0][0]

    print(entidad_+1, genero_, nivelSalarial_+1, organismo_+1, 1, funcion_+1)

    XGB_abuso = np.array([[entidad_+1, genero_, nivelSalarial_+1, organismo_+1, 1, funcion_+1]])  # Entrada de datos al modelo

    #Convertir el caso a formato DMatrix
    dcase = xgb.DMatrix(data=XGB_abuso)

    #Realizar la predicción
    prediccion_prob = model1.predict(dcase)[0]  # Probabilidad para la clase 1
    prediccion = 1 if prediccion_prob > 0.5 else 0  # Convertir probabilidad a clase

    probabilid_edos = []
    for i in range(1, 33):
        # Convertir el caso actual a formato DMatrix
        caso_xgb = xgb.DMatrix(np.array([[i, genero_, nivelSalarial_+1, organismo_+1, 1, funcion_+1]]))

        # Obtener la probabilidad de la clase positiva (configuración de `predict` con Booster)
        probabilidad_abuso_estado = model1.predict(caso_xgb)[0]  # Booster predice directamente probabilidades

        # Almacenar resultado en la lista
        probabilid_edos.append({
            "CVE_ENT": i,  # El primer valor representa la entidad federativa
            "Metrica": probabilidad_abuso_estado
        })

    # Convertir los resultados a un DataFrame
    df_edos_abuso = pd.DataFrame(probabilid_edos)
    df_edos_abuso.insert(loc=1, column='entidad_federativa', value=map_entidad_federativa)

    #Mostrar los resultados
    if prediccion == 1:
        st.error("El modelo XGBoost predice que el servidor público realizará abuso sexual")
    else:
        st.warning("El modelo XGBoost predice que el servidor público podría realizar abuso sexual")
    
    if prediccion_prob > 0.5:
        st.error(f"La probabilidad de que el servidor público realice abuso sexual: {prediccion_prob:.2f}")
    else:
        st.warning(f"La probabilidad de que el servidor público realice abuso sexual: {prediccion_prob:.2f}")

# Unir datos al GeoDataFrame
gdf = pd.read_csv("mexico_geolocalizacion.csv")
gdf = gdf.merge(df_edos_abuso, left_on="entidad_federativa", right_on="entidad_federativa", how="left")

# Crear colores basados en la métrica
def get_color(value):
    if value < 0.50:
        return "#008f39"
    elif value < 0.80:
        return "#ffff00"
    else:
        return "#ff0000"

gdf["color"] = gdf["Metrica"].apply(get_color)

st.map(gdf, latitude="latitud", longitude="longitud", size=100, color="color")
df_filtered = df_intel[df_intel['CVE_ENT'] == entidad_+1]
st.write(df_filtered)