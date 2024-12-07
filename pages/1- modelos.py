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
    'Distrito Federal',
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
st.sidebar.title("Abuso de Autoridad y Abuso Sexual")
st.sidebar.markdown("Escribe aquí tus preferencias:")

entidad = st.sidebar.selectbox("Entidad federativa", map_entidad_federativa)
organismo = st.sidebar.selectbox("Organismo de poder", map_organismo_poder)
funcion = st.sidebar.selectbox("Función de la institución", map_funcion_inst)
nivelSalarial = st.sidebar.selectbox("Nivel salarial", map_nivel_salarial)
genero = st.sidebar.selectbox("Género", map_genero)

if models == "Random Forest | Abuso de autoridad":
    entidad_ = np.where(map_entidad_federativa == entidad)[0][0]
    genero_ = np.where(map_genero == genero)[0][0]
    nivelSalarial_ = np.where(map_nivel_salarial == nivelSalarial)[0][0]
    organismo_ = np.where(map_organismo_poder == organismo)[0][0]
    funcion_ = np.where(funcion == funcion)[0][0]

    caso_rfc = np.array([[entidad_+1, genero_, nivelSalarial_+1, organismo_+1, 1, funcion_+1]])

    prediccion = model2.predict(caso_rfc)
    probabilidad_abuso_autoridad = model2.predict_proba(caso_rfc)[0, 1]  # Probabilidad de default (clase 1)

    if prediccion[0] == 1:
        st.success("El modelo predice que el servidor público realiza algún tipo de Abuso de autoridad")
    else:
        st.success("El modelo predice que el servidor público no realizará algún tipo de Abuso de autoridad")
    
    st.write(f"La probabilidad de que el servidor público realice un tipo de abuso de autoridad es: {probabilidad_abuso_autoridad:.2f}")
else :
    entidad_ = np.where(map_entidad_federativa == entidad)[0][0]
    genero_ = np.where(map_genero == genero)[0][0]
    nivelSalarial_ = np.where(map_nivel_salarial == nivelSalarial)[0][0]
    organismo_ = np.where(map_organismo_poder == organismo)[0][0]
    funcion_ = np.where(funcion == funcion)[0][0]

    print(entidad_+1, genero_, nivelSalarial_+1, organismo_+1, 1, funcion_+1)

    XGB_abuso = np.array([[entidad_+1, genero_, nivelSalarial_+1, organismo_+1, 1, funcion_+1]])  # Entrada de datos al modelo

    # 3. Convertir el caso a formato DMatrix
    dcase = xgb.DMatrix(data=XGB_abuso)

    # 4. Realizar la predicción
    prediccion_prob = model1.predict(dcase)[0]  # Probabilidad para la clase 1
    prediccion = 1 if prediccion_prob > 0.5 else 0  # Convertir probabilidad a clase

    # 5. Mostrar los resultados
    if prediccion == 1:
        st.success("El modelo predice que el servidor público realizará abuso sexual")
    else:
        st.success("El modelo predice que el servidor público no realizará abuso sexual")
    
    st.write(f"La probabilidad de que el servidor público realice abuso sexual: {prediccion_prob:.2f}")

# Filter the movies based on the user input
df_filtered = df_intel[df_intel['CVE_ENT'] == entidad_+1]
st.write(df_filtered)