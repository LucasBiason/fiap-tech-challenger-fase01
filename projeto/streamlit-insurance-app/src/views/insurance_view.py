import streamlit as st
import requests

def insurance_form():
    st.title("Previsão de Custos")
    st.write("Previsão de custos de saúde com base em dados do paciente.")
    st.write("Preencha os dados abaixo para obter uma previsão.")
    st.write("Os dados são os seguintes:")
    with st.form(key='insurance_form'):
        smoker = st.selectbox("Fumante?", ["yes", "no"])
        age = st.number_input("Idade", min_value=0, max_value=120)
        bmi = st.number_input("IMC", min_value=0.0, format="%.2f")
        children = st.number_input("Número de Filhos", min_value=0)

        submit_button = st.form_submit_button(label='Enviar')

        if submit_button:
            data = {
                "smoker": smoker,
                "age": age,
                "bmi": bmi,
                "children": children
            }
            response = requests.post("http://localhost:8004/predict", json=data)
            if response.status_code == 200:
                prediction = response.json()
                st.success(f"Custo previsto: $ {prediction['charge']:.2f}")
            else:
                st.error("Erro ao fazer a previsão.")