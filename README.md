# Tech Challenger 

Este projeto é um desafio técnico feito como conclusão do primeiro módulo do curso Pós Tech - IA para Devs para a FIAP.
Consiste na construção de um módelo preditivo de uma base de dados utilizando so conhecimentos do curso.

## Projeto de Análise

Esta fase consiste em escolher a base de dados que iremos trabalha, fazer a exploração das caracteristicas assim como uma analise dos dados. Após isso construir o modelo de previsão dos custos médicos.

Os arquivos referentes a esta fase estão na pasta **analise**.

### Etapa 01 - Análise e Preparação dos Dados

Nesta etapa estamos fazendo analise do conjunto de dados para aprender o máximo possível sobre ele, assim como o que podemos extrair de informações.
O foco é em preparar os dados para os modelos de dados.

### Etapa 02 - Construção do Modelo Preditivo

Nesta etapa vamos aproveitar a limpeza dos dados e tratamento de valores que foi feita na etapa anterior e criar um modelo preditivo de regressão utilizando algumas técnicas e verificar o que fica melhor para a distruição dos dados.


## Projeto Entregável

Esta fase consiste em criar um pacote utilizavel com o modelo preditivo realizado na primeira parte do desafio e apresentar os resultados obtivos.

Os arquivos referentes a esta fase estão na pasta **projeto**.


### Instalação para o ambiente

1. Crie um ambiente virtual:
```bash
virtualenv .venv 
source .venv/bin/activate
```
2. Faça a instalação das bibliotecas:
```bash
pip install -r requirements.txt
```
3. Execute o projeto da API Predict Service:
```bash
make run-dev
```
4. Execute o projeto do frontend:
```bash
make run-front-dev
```


#### Exemplo de requisição para o endpoint `/predict`

**Usando curl:**
```bash
curl -X POST "http://localhost:8004/predict" \
  -H "Content-Type: application/json" \
  -d '{"smoker": "no", "age": 33, "bmi": 25, "children": 0}'
```

**Usando python:**
```python
import requests

url = "http://localhost:8004/predict"
data = {
    "smoker": "no",
    "age": 33,
    "bmi": 25,
    "children": 0
}

response = requests.post(url, json=data)
print(response.json())
```