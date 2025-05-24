import streamlit as st

st.set_page_config(page_title="Tech Challenger", layout="wide")


st.sidebar.title("Navegação")
page = st.sidebar.radio(
    "Selecione uma página:", 
    ["Análise e Processamento", "Construção do Modelo", "Previsão de Custos"]
)

if page == "Análise e Processamento":
    from views.analysis_view import analysis_notebook
    analysis_notebook()
elif page == "Construção do Modelo":
    from views.model_view import model_notebook
    model_notebook()
elif page == "Previsão de Custos":
    from views.insurance_view import insurance_form
    insurance_form()