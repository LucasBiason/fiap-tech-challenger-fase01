### Estrutura do Projeto

```plaintext
my_streamlit_app/
│
├── app.py                   # Arquivo principal da aplicação Streamlit
├── pages/                   # Pasta para as páginas da aplicação
│   ├── page1.py             # Página 1: Análise e Processamento
│   ├── page2.py             # Página 2: Construção do Modelo
│   └── page3.py             # Página 3: Formulário de Entrada de Dados
│
├── requirements.txt         # Dependências do projeto
└── utils/                   # Pasta para funções utilitárias
    ├── load_notebook.py     # Função para carregar e exibir o conteúdo dos notebooks
    └── api_request.py       # Função para fazer requisições à API FastAPI
```

### 1. Criar o arquivo `app.py`

```python
import streamlit as st

st.set_page_config(page_title="Insurance Cost Prediction", layout="wide")

# Menu lateral
st.sidebar.title("Menu")
page = st.sidebar.radio("Selecione uma página:", ["Análise e Processamento", "Construção do Modelo", "Formulário de Entrada de Dados"])

if page == "Análise e Processamento":
    from pages.page1 import show_page1
    show_page1()
elif page == "Construção do Modelo":
    from pages.page2 import show_page2
    show_page2()
elif page == "Formulário de Entrada de Dados":
    from pages.page3 import show_page3
    show_page3()
```

### 2. Criar a página 1 (`pages/page1.py`)

```python
import streamlit as st
from utils.load_notebook import load_notebook

def show_page1():
    st.title("Análise e Processamento")
    st.write("Conteúdo do arquivo 'etapa-01-analise-processamento.ipynb':")
    load_notebook("etapa-01-analise-processamento.ipynb")
```

### 3. Criar a página 2 (`pages/page2.py`)

```python
import streamlit as st
from utils.load_notebook import load_notebook

def show_page2():
    st.title("Construção do Modelo")
    st.write("Conteúdo do arquivo 'etapa-02-modelo.ipynb':")
    load_notebook("etapa-02-modelo.ipynb")
```

### 4. Criar a página 3 (`pages/page3.py`)

```python
import streamlit as st
from utils.api_request import make_api_request

def show_page3():
    st.title("Formulário de Entrada de Dados")
    
    # Formulário para entrada de dados
    with st.form(key='insurance_form'):
        smoker = st.selectbox("Fumante:", ["sim", "não"])
        age = st.number_input("Idade:", min_value=0, max_value=120)
        bmi = st.number_input("IMC:", min_value=0.0, format="%.2f")
        children = st.number_input("Número de Filhos:", min_value=0)

        submit_button = st.form_submit_button(label='Enviar')

        if submit_button:
            data = {
                "smoker": smoker,
                "age": age,
                "bmi": bmi,
                "children": children
            }
            result = make_api_request(data)
            st.write(f"Custo do seguro previsto: ${result}")
```

### 5. Criar a função para carregar o notebook (`utils/load_notebook.py`)

```python
import streamlit as st
import nbformat
from nbconvert import MarkdownExporter

def load_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)
    
    # Converter o notebook para Markdown
    exporter = MarkdownExporter()
    body, _ = exporter.from_notebook_node(notebook_content)
    
    # Exibir o conteúdo no Streamlit
    st.markdown(body)
```

### 6. Criar a função para fazer requisições à API (`utils/api_request.py`)

```python
import requests

def make_api_request(data):
    url = "http://localhost:8004/predict"  # URL da API FastAPI
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        return response.json().get("prediction", "Erro ao obter previsão.")
    else:
        return "Erro ao fazer requisição à API."
```

### 7. Criar o arquivo `requirements.txt`

```plaintext
streamlit
nbformat
nbconvert
requests
```

### 8. Executar a aplicação

Para executar a aplicação, navegue até o diretório do projeto e execute o seguinte comando:

```bash
streamlit run app.py
```

### Considerações Finais

- Certifique-se de que a API FastAPI esteja rodando na porta 8004 antes de testar a aplicação.
- Você pode personalizar ainda mais a interface do Streamlit, adicionando mais estilos e funcionalidades conforme necessário.
- A função `load_notebook` converte o conteúdo do Jupyter Notebook para Markdown, permitindo que o conteúdo seja exibido de forma interativa no Streamlit.