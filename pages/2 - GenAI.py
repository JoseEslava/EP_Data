import streamlit as st
import numpy as np

from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="EPData GenAI", page_icon="bar-chart")
st.header('Analítica aplicando Inteligencia Artificial Generativa')

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

map_tipo_delito = np.array([
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

# Create a sidebar for user input
st.sidebar.title("Conversa con los datos")
st.sidebar.markdown("Escribe aquí tus preferencias:")

entidad = st.sidebar.selectbox("Entidad federativa", map_entidad_federativa)
delito = st.sidebar.selectbox("Tipo delito", map_tipo_delito)

loader = CSVLoader(file_path="anticorrupcion_intel.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()

text_splitter = CharacterTextSplitter(separator='\n')
splits = text_splitter.split_documents(data)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

template_prefix = """Tu eres un investigador de la corrupción en México y quieres encontrar patrones en los datos. 
Usa el siguiente contexto para responder la pregunta que se encuentra al final. 
En caso de no sber la respuesta, contesta con "No sé la respuesta". No busques información adicional.

{context}"""
user_info = """Esta es la información que sabes para encontrar patrones, usa esta infrmación para mejorar tu búsqueda:
Entidad Federativa: {CVE_ENT}
Delito: {delito}"""

template_suffix= """Pregunta: {question}
Tu respuesta:"""

info_ = user_info.format(CVE_ENT = entidad, delito = delito)

COMBINED_PROMPT = template_prefix +'\n'+ info_ +'\n'+ template_suffix

#setting up the chain
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

PROMPT = PromptTemplate(
    template=COMBINED_PROMPT, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever,
                                 return_source_documents=True,
                                 chain_type_kwargs=chain_type_kwargs)

query = st.text_input('¿Qué estás buscando en la plataforma EP Data?', placeholder = '¿Qué patrones de corrupción encuentras con las características seleccionadas?')
if query:
    result = qa({"query": query})
    st.write(result['result'])
