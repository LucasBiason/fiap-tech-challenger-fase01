import os
import streamlit as st
import nbformat
from nbconvert import HTMLExporter

def model_notebook():
    st.title("Construção do Modelo")

    # Carregar o notebook
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    notebook_path = os.path.join(project_dir, "src/views/notebooks", "etapa-02.ipynb")
    with open(notebook_path) as f:
        notebook_content = nbformat.read(f, as_version=4)

    # Converter para HTML
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(notebook_content)

    # Exibir o conteúdo HTML
    st.components.v1.html(body, height=800, scrolling=True)