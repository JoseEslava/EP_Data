import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EPData KPIs", page_icon="bar-chart")
st.header('Comparativa de Organismos, Causas y Funciones')

# Datos de entrada
df_organismo = pd.read_csv('organismo.csv', sep=',', header=0)
df_causa_motivo = pd.read_csv('causaMotivo.csv', sep=',', header=0)
df_funcion_institucion = pd.read_csv('funcionInstitucion.csv', sep=',', header=0)

with st.container():
    st.subheader('Porcentaje de Casos por Tipo de Organismo')

    # Gráfico de pie
    fig = px.pie(
        df_organismo, values='Metrica', names='organismo_poder', 
        title='Porcentaje de Casos por Tipo de Organismo'
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.1])
    fig.update_layout(height=500, width=500)
    st.plotly_chart(fig, use_container_width=True)

    st.write(df_organismo)

with st.container():
    st.subheader('Total de Casos por Causa o Motivo')
    # Gráfico de barras 
    st.bar_chart(df_causa_motivo, y='Total_casos', color='Categoria')
    # Tabla de datos
    st.write(df_causa_motivo)

with st.container():
    st.subheader('Dispersión de Casos por Función e Institución')

    # Gráfico de dispersión en la tercera columna con Plotly
    fig = px.scatter(
        df_funcion_institucion, x='Funcion_inst', y='Total_casos', color='Metrica',
        title='Dispersión de Casos por Función e Institución', size='Total_casos', hover_name='Funcion_inst'
    )
    fig.update_layout(xaxis_title='', yaxis_title='Total de Casos', xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.write(df_funcion_institucion)