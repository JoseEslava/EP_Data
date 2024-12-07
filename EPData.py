import streamlit as st

st.set_page_config(page_title="EPData Anticorrupcion", page_icon="bar-chart")
      
st.header("Bienvenido a la plataforma 'EP Data' de alertamiento Anti-corrupción")
st.sidebar.success("Selecciona alguna de las opciones")

st.markdown(
    """
    EP Data es un proyecto de alertamiento Anti-corrupción enfocado en
    - Abuso de autoridad
    - Abuso sexual
    
    Implementa algoritmos de Machine Learning e Inteligencia Artificial Generativa para
    predecir posibles casos de corrupción en servidores públicos, a través de una
    caracterización definida por las siguientes varibales:
    - Función institucional
    - Autoridad sancionadora
    - Tipo de organismo
    - Nivel salarial
    - Género
    - Entidad federativa

    **👈 Selecciona una funcionalidad ... haciendo clic en el sidebar**    
"""
)