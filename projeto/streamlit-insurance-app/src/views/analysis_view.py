import streamlit as st
import nbformat
from nbconvert import HTMLExporter
import os

def analysis_notebook():
    st.title("Análise e Processamento")

    # Carregar o notebook
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    notebook_path = os.path.join(project_dir, "src/views/notebooks", "etapa-01.ipynb")
    with open(notebook_path) as f:
        notebook_content = nbformat.read(f, as_version=4)

    # Converter para HTML
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(notebook_content)

    # Exibir o conteúdo HTML
    st.components.v1.html(body, height=800, scrolling=True)