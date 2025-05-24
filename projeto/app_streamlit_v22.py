import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import io

st.set_page_config(page_title='Análise de Custos Médicos', layout='wide')
st.title('Tech Challenger - Análise de Custos Médicos')
st.markdown('''Como profissional responsável pelo desenvolvimento de um modelo preditivo de regressão para estimar os custos médicos individuais cobrados por seguros de saúde, utilizamos o conjunto de dados [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance/data) para validar hipóteses relacionadas aos fatores que impactam a definição dos custos dos planos de saúde.''')

st.subheader('Objetivo do Projeto')
st.markdown('''Nosso objetivo é analisar a estrutura dos dados, realizar a limpeza necessária e explorar questionamentos sobre os insights que podemos extrair a partir deles.''')

# Carregamento dos dados (substituir pelo caminho ou método adequado)
data = pd.read_csv('../analise/dados/insurance.csv')  
st.subheader('Estrutura e Qualidade dos Dados')

# Exibir shape e tipos das colunas (substituindo data.info())

buffer = io.StringIO()
data.info(buf=buffer)
# st.text(buffer.getvalue())

# Alternativamente: mostrar tipos e amostra dos dados
st.write("Dimensões do dataset:", data.shape)
st.write("Tipos de dados:")
st.write(data.dtypes)
st.subheader("Prévia dos dados")
st.dataframe(data.head())

st.subheader("Valores Nulos por Coluna")

null_counts = data.isnull().sum()

if null_counts.sum() == 0:
    st.success("Não há valores nulos no dataset - dados completos!")
else:
    st.warning("Existem valores nulos nas colunas abaixo:")
    st.dataframe(null_counts[null_counts > 0])


st.markdown("""
O conjunto de dados contém **1338 entradas** e **7 colunas**.

### Estrutura das colunas:
- `age`: idade
- `sex`: sexo
- `bmi`: índice de massa corporal (IMC)
- `children`: número de filhos
- `smoker`: fumante ou não
- `region`: região onde reside
- `charges`: custos médicos

### Qualidade dos dados:
- **Não há valores ausentes**.
- **Variáveis categóricas**: `sex`, `smoker`, `region`
- **Variáveis numéricas**: `age`, `children`, `bmi`, `charges`
            
- A Variável Dependente do conjunto de dados é a coluna: 'charges', representando os custos médicos.
- A maioria das idades está bem distribuída entre 18 e 64 anos, sem grandes concentrações em uma faixa específica.
- A maioria das pessoas tem BMI (Índice de Massa Corporal) entre 20 e 40, com poucos casos acima de 40 (possíveis outliers).
- Valores altos de BMI indicam que há pessoas obesas no conjunto de dados.
- A maioria das pessoas possui entre 0 e 2 filhos, sendo que o valor 0 (sem filhos) é o mais frequente. Apenas uma pequena parcela da população tem 3 ou mais filhos.
- A maioria dos custos médicos está concentrada em valores mais baixos, mas há alguns casos com custos muito altos (outliers).            
""")

st.subheader('Distribuições Iniciais')
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(data['age'], kde=True, ax=axs[0,0], color='skyblue')
axs[0,0].set_title('Distribuição de Idades')
sns.histplot(data['bmi'], kde=True, ax=axs[0,1], color='orange')
axs[0,1].set_title('Distribuição de IMC')
sns.histplot(data['children'], bins=6, ax=axs[1,0], color='green')
axs[1,0].set_title('Distribuição de Filhos')
sns.histplot(data['charges'], kde=True, ax=axs[1,1], color='red')
axs[1,1].set_title('Distribuição de Custos Médicos')
plt.tight_layout(),
st.pyplot(fig)
plt.clf()

st.subheader('Análise e Visualização dos Dados')
st.markdown('Com isso, levantamos algumas questões relevantes sobre os dados analisados:')

st.subheader('Pergunta 01: Existem diferenças nos custos médios entre as regiões?')
st.markdown('''Não há diferenças significativas nos custos entre as regiões. As medianas dos custos são bastante próximas, indicando que a localização geográfica não é um fator determinante para os custos médicos.''')
sns.boxplot(x='region', y='charges', data=data, hue="region")
plt.title('Custos Médicos por Região')
plt.xlabel('Região')
plt.ylabel('Custos Médicos ($)')
st.pyplot(plt.gcf())
plt.clf()

st.subheader('Pergunta 02: O hábito de fumar impacta os custos médicos? Fumantes têm custos médicos mais altos ou mais baixos?')
st.markdown('''Fumantes têm custos médicos significativamente mais altos em comparação com não fumantes, com isso sabemos que o hábito de fumar é um dos principais fatores que influenciam nos custos do plano de saúde.''')
import matplotlib.pyplot as plt
import seaborn as sns

# Contagem de fumantes e não fumantes
smoker_counts = data['smoker'].value_counts()
labels_smoker = [label.title() for label in smoker_counts.index]

# Criar figura com 2 subplots lado a lado
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico de Pizza
axs[0].pie(smoker_counts, labels=labels_smoker, autopct='%1.1f%%', startangle=90, colors=["#1c5e95","#e6b72a"])
axs[0].set_title('Distribuição de Fumantes')

# Boxplot dos custos médicos
sns.boxplot(x='smoker', y='charges', data=data, ax=axs[1])
axs[1].set_title('Custos Médicos por Hábito de Fumar')
axs[1].set_xlabel('Fumante')
axs[1].set_ylabel('Custos Médicos')

# Layout ajustado
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

st.subheader('Pergunta 03: O sexo influencia os gastos com saúde?')
st.markdown('''Não há diferenças significativas nos custos médicos entre homens e mulheres. As distribuições são bastante semelhantes, mesmo com os valores indicando um custo maximo maior para os pacientes do sexo masculino, talvez o sexo seja é um fator tão relevante para os custos médicos. Vamos precisar verificar melhor ao construir o modelo de previsão.''')
import matplotlib.pyplot as plt
import seaborn as sns

# Contagem por sexo
sex_counts = data['sex'].value_counts()
labels_sex = [label.title() for label in sex_counts.index]

# Criar uma figura com 3 subplots organizados em 1 linha e 3 colunas
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Gráfico de Pizza - Distribuição por Sexo
axs[0].pie(sex_counts, labels=labels_sex, autopct='%1.1f%%', startangle=90, colors=["#1c5e95","#e6b72a"])
axs[0].set_title('Distribuição da População por Sexo')

# Boxplot - Custos Médicos por Sexo
sns.boxplot(x='sex', y='charges', data=data, ax=axs[1])
axs[1].set_title('Custos Médicos por Sexo')
axs[1].set_xlabel('Sexo')
axs[1].set_ylabel('Custos Médicos')

# Boxplot - Custos Médicos por Sexo e Hábito de Fumar
sns.boxplot(x='sex', y='charges', hue='smoker', data=data, ax=axs[2])
axs[2].set_title('Custos Médicos por Sexo e Hábito de Fumar')
axs[2].set_xlabel('Sexo')
axs[2].set_ylabel('Custos Médicos')
axs[2].legend(title='Fumante')

# Ajustar layout
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

st.subheader('Pergunta 04: O IMC (índice de massa corporal), presente no campo BMI dos dados, tem impacto nos custos médicos?')
st.markdown('''Há uma correlação entre IMC e custos médicos, especialmente para os casos de fumantes. Assim podemos verificar que fumantes com IMC elevado tendem a ter custos médicos mais altos, enquanto não fumantes apresentam uma relação mais dispersa. Podemos inferir que as duas variáveis juntas tem alto impacto nos custos, logo iremos utilizar as duas no modelo.''')
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=data)
plt.title('Relação entre IMC e Custos Médicos')
plt.xlabel('IMC')
plt.ylabel('Custos Médicos')
st.pyplot(plt.gcf())
plt.clf()

st.subheader('Pergunta 05: Como os custos médicos variam com a idade?')
st.markdown('''Os custos médicos aumentam com a idade, indicando que pessoas mais velhas tendem a ter despesas médicas mais altas. Isso pode estar relacionado ao aumento de problemas de saúde com o envelhecimento, sendo assim a idade é uma das variáveis que vamos utilizar no nosso modelo.''')
import matplotlib.pyplot as plt
import seaborn as sns

# Criar a figura com 2 subplots lado a lado
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1 - Custos Médicos por Idade (Geral)
sns.scatterplot(data=data, x="age", y="charges", alpha=0.6, ax=axs[0])
axs[0].set_title("Custos Médicos por Idade")
axs[0].set_xlabel("Idade")
axs[0].set_ylabel("Custos Médicos ($)")
axs[0].grid(True, linestyle='--', alpha=0.3)

# Gráfico 2 - Custos Médicos por Idade e Hábito de Fumar
sns.scatterplot(data=data, x="age", y="charges", hue="smoker", alpha=0.6, ax=axs[1])
axs[1].set_title("Custos Médicos por Idade e Fumantes")
axs[1].set_xlabel("Idade")
axs[1].set_ylabel("Custos Médicos ($)")
axs[1].grid(True, linestyle='--', alpha=0.3)
axs[1].legend(title="Fumante")

# Ajustar layout
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

st.subheader('Pergunta 06: A presença de filhos altera significativamente os gastos médicos?')
st.markdown('''O número de filhos exerce influência nos custos médicos. No entanto, observa-se que, a partir de cinco filhos, há uma redução nos custos médios. Apesar disso, não é possível afirmar que o acréscimo de dados manteria essa tendência de redução nos valores dos planos devido a falta de informações na base analisada.
## Pré Processamento dos Dados
Nessa etapa nós iremos realizar um processamento dos dados para que eles possam ser colocados no modelo preditivo.
Conforme vimos na análise e visualiação de dados, temos o seguinte:
* Como não existem valores nulos, não precisaremos realizar preenchimento dos mesmos, o que no caso poderia ser com a mediana dos valores.
* Temos colunas categóricas que precisam ser processadas para entrar no modelo. São elas: **sex**, **smoker** e **region**. Utilizaremos um **One-Hot Encoding**.
* Vamos remover os outliers possiveis dos dados para evitar discrepancias enormes.
* A Variavel dependente a ser prevista é a coluna **charges**.
* As colunas com maior potencial de alteração no modelo são: **smoker**, **age** e **bmi**.
### Transformação dos Dados Categoricos
Para criação do modelo não podemos utilizar as colunas categoricas como elas estão, em forma de texto. Precisamos transforma-las em números.
Não podemos colocar qualquer número, por isso precisamos criar novas colunas a partir das categoricas e para cada uma, a nova coluna tera um valor correspondente a 1 ou 0 para indicar a presença da caracteristica. Isso é o que chamamos de One Hot Encoding.

No exemplo desses dados, a coluna smoker, apresenta dois dados possiveis:
Pelo One Hot Encoding, iremos construir duas novas colunas: smoker_yes e smoker_no.
onde smoker_yes recebe 1 para quem é fumante e 0 para quem não tem a caracteristica. O contrario para a coluna smoker_no.
Faremos isso para todos os dados categoricos do conjunto.
### Análise de Outliers
Outlier é um item que se desvia significativamente do restante dos itens. Identificar outliers é importante em estatística e análise de dados, pois eles podem ter um impacto significativo nos resultados de análises estatísticas, distorcendo estatisticas.
Eles podem ser causados por erros de medição, amostragem, erros experimentais ou ate mesmo amostragens de multiplas populações.
Nós temos dois metodos de remoção:

O **IQR (Interquartile Range)**, ou intervalo interquartil, é uma técnica estatística usada para identificar e remover outliers (valores extremos) em um conjunto de dados. Ele é baseado nos quartis, que dividem os dados em quatro partes iguais.

Ele é robusto contra valores extremos, pois se baseia nos quartis e não na média.
Ajuda a limpar os dados, **removendo valores que podem distorcer análises ou modelos preditivos**.
O **Z-Score** é uma métrica estatística que mede o quão distante um valor está da média em termos de desvios padrão.
Para cada valor no conjunto de dados, o Z-Score é calculado como:  

(valor do dado - média do conjunto de dados) / desvio padrão do conjunto de dados.

* Um Z-Score de 0 significa que o valor está exatamente na média.
* Valores positivos indicam que o dado está acima da média, enquanto valores negativos indicam que está abaixo.
* Quanto maior (ou menor) o Z-Score, mais distante o valor está da média.
* Valores com Z-Scores muito altos ou muito baixos (geralmente acima de 3 ou abaixo de -3) são considerados outliers, pois estão muito distantes da média.

Ele é útil para detectar outliers em dados que seguem uma distribuição aproximadamente normal.
É uma abordagem simples e eficaz para identificar valores extremos em relação à média e ao desvio padrão.
### VALIDAR ESSA VISAO
### Matriz de Correlação
A matriz de correlação é uma tabela que mostra as correlações entre diferentes variáveis em um conjunto de dados. A correlação mede a força e a direção do relacionamento linear entre duas variáveis.

Valores de correlação variam de -1 a 1.
* 1: Correlação positiva perfeita (quando uma variável aumenta, a outra também aumenta).
* -1: Correlação negativa perfeita (quando uma variável aumenta, a outra diminui).
* 0: Nenhuma correlação (as variáveis não têm relação linear).

Correlação próxima de 1 ou -1 indica uma relação forte.

Correlação próxima de 0 indica uma relação fraca ou inexistente.
* **smoker_yes** tem correlação positiva e negativa com **charges**. Isso indica que ser fumante é o principal fator que aumenta os custos médicos.
* **age** também tem uma correlação positiva significativa, sugerindo que a idade é outro fator importante.
* **bmi** tem uma correlação mais fraca, mas ainda pode ser relevante.
* **children** e **region_northwest** têm uma correlação moderada.
Nessa etapa nós iremos realizar um processamento dos dados para que eles possam ser colocados no modelo preditivo.
Conforme vimos na análise e visualiação de dados, temos o seguinte:
* Como não existem valores nulos, não precisaremos realizar preenchimento dos mesmos, o que no caso poderia ser com a mediana dos valores.
* Temos colunas categóricas que precisam ser processadas para entrar no modelo. São elas: **sex**, **smoker** e **region**. Utilizaremos um **One-Hot Encoding**.
* Vamos remover os outliers possiveis dos dados para evitar discrepancias enormes.
* A Variavel dependente a ser prevista é a coluna **charges**.
* As colunas com maior potencial de alteração no modelo são: **smoker**, **age** e **bmi**.
### Transformação dos Dados Categoricos
Para criação do modelo não podemos utilizar as colunas categoricas como elas estão, em forma de texto. Precisamos transforma-las em números.
Não podemos colocar qualquer número, por isso precisamos criar novas colunas a partir das categoricas e para cada uma, a nova coluna tera um valor correspondente a 1 ou 0 para indicar a presença da caracteristica. Isso é o que chamamos de One Hot Encoding.

No exemplo desses dados, a coluna smoker, apresenta dois dados possiveis:
Pelo One Hot Encoding, iremos construir duas novas colunas: smoker_yes e smoker_no.
onde smoker_yes recebe 1 para quem é fumante e 0 para quem não tem a caracteristica. O contrario para a coluna smoker_no.
Faremos isso para todos os dados categoricos do conjunto.
### Análise de Outliers
Outlier é um item que se desvia significativamente do restante dos itens. Identificar outliers é importante em estatística e análise de dados, pois eles podem ter um impacto significativo nos resultados de análises estatísticas, distorcendo estatisticas.
Eles podem ser causados por erros de medição, amostragem, erros experimentais ou ate mesmo amostragens de multiplas populações.
Nós temos dois metodos de remoção:

O **IQR (Interquartile Range)**, ou intervalo interquartil, é uma técnica estatística usada para identificar e remover outliers (valores extremos) em um conjunto de dados. Ele é baseado nos quartis, que dividem os dados em quatro partes iguais.

Ele é robusto contra valores extremos, pois se baseia nos quartis e não na média.
Ajuda a limpar os dados, **removendo valores que podem distorcer análises ou modelos preditivos**.
O **Z-Score** é uma métrica estatística que mede o quão distante um valor está da média em termos de desvios padrão.
Para cada valor no conjunto de dados, o Z-Score é calculado como:  

(valor do dado - média do conjunto de dados) / desvio padrão do conjunto de dados.

* Um Z-Score de 0 significa que o valor está exatamente na média.
* Valores positivos indicam que o dado está acima da média, enquanto valores negativos indicam que está abaixo.
* Quanto maior (ou menor) o Z-Score, mais distante o valor está da média.
* Valores com Z-Scores muito altos ou muito baixos (geralmente acima de 3 ou abaixo de -3) são considerados outliers, pois estão muito distantes da média.

Ele é útil para detectar outliers em dados que seguem uma distribuição aproximadamente normal.
É uma abordagem simples e eficaz para identificar valores extremos em relação à média e ao desvio padrão.
### VALIDAR ESSA VISAO
### Matriz de Correlação
A matriz de correlação é uma tabela que mostra as correlações entre diferentes variáveis em um conjunto de dados. A correlação mede a força e a direção do relacionamento linear entre duas variáveis.

Valores de correlação variam de -1 a 1.
* 1: Correlação positiva perfeita (quando uma variável aumenta, a outra também aumenta).
* -1: Correlação negativa perfeita (quando uma variável aumenta, a outra diminui).
* 0: Nenhuma correlação (as variáveis não têm relação linear).

Correlação próxima de 1 ou -1 indica uma relação forte.

Correlação próxima de 0 indica uma relação fraca ou inexistente.
* **smoker_yes** tem correlação positiva e negativa com **charges**. Isso indica que ser fumante é o principal fator que aumenta os custos médicos.
* **age** também tem uma correlação positiva significativa, sugerindo que a idade é outro fator importante.
* **bmi** tem uma correlação mais fraca, mas ainda pode ser relevante.
* **children** e **region_northwest** têm uma correlação moderada.
## Processamento do Modelo
Como as colunas 'Região' e 'Sexo' não contribuem diretamente para a construção do modelo preditivo, elas serão removidas do conjunto de dados
Aplicando a 
[Tabela de Obesidade](https://www.cancer.org/cancer/risk-prevention/diet-physical-activity/body-weight-and-cancer-risk.html), onde:
* IMC entre 25,0 e 29,9 Kg/m2: sobrepeso; 
* IMC entre 30,0 e 34,9 Kg/m2: obesidade grau I; 
* IMC entre 35,0 e 39,9 Kg/m2: obesidade grau II; 
* IMC maior do que 40,0 Kg/m2: obesidade grau III.
## Construção do Modelo

Nessa etapa vamos dividir os dados de teste e treinamento e escolher os modelos que podemos utilizar para previsão.
Como avaliar o que está bom ou ruim:
* MSE e MAE:
    * Compare os valores entre os modelos. O modelo com os menores valores de MSE e MAE é o mais preciso.
    * Se os valores forem muito altos, o modelo pode estar subajustado (underfitting).
* R²:
    * Um valor próximo de 1 é desejável.
    * Se o R² for muito baixo, o modelo pode não estar capturando bem os padrões dos dados.
    * Se for muito alto, verifique se o modelo não está superajustado (overfitting).
* QQ Plot:
    * Resíduos alinhados indicam que o modelo está atendendo à suposição de normalidade.
    * Se houver desvios, considere ajustar o modelo ou revisar os dados.
* MAPE:
    * ???
### Selecionando alguns modelos e verificando o desempenho
Monstrando os atributos que tem relevancia para o modelo
### Análise do Gráfico SHAP - Importância das Variáveis

O gráfico SHAP apresenta a contribuição de cada variável na previsão dos custos médicos, destacando como cada feature impacta positiva ou negativamente o resultado do modelo.

---

### **Como interpretar o gráfico:**
- **Eixo Y:** Lista das variáveis, ordenadas da mais impactante (topo) para a menos relevante (base).
- **Eixo X:** Mostra o valor SHAP, ou seja, o **impacto da variável na previsão**:
  - Valores positivos → **aumentam** o custo previsto.
  - Valores negativos → **diminuem** o custo previsto.
- **Cada ponto:** Representa um registro da base de dados:
  - **Azul:** Valor baixo da variável.
  - **Rosa/vermelho:** Valor alto da variável.

---

### **Principais insights do modelo:**

- **`smoker_yes` (Fumante)**  
  → Fumar tem o maior impacto no custo.  
  → Pessoas que fumam (vermelho) puxam fortemente o custo **para cima**.  
  → Não fumantes (azul) puxam o custo **para baixo**.

- **`age` (Idade)**  
  → Idades mais altas (vermelho) estão associadas a custos mais altos.  
  → Idades mais baixas (azul) tendem a reduzir o custo.

- **`bmi` (Índice de Massa Corporal)**  
  → BMI mais alto (vermelho) contribui para aumento dos custos.  
  → BMI mais baixo (azul) tende a reduzir os custos.

- **`children` (Número de filhos)**  
  → Tem impacto moderado no custo, com variações menores.

- **`sex_male` (Sexo masculino)** e **`region_*` (Região)**  
  → Impacto relativamente pequeno, mas ainda contribuem para o modelo.

- **`weight_condition_*` (Condição de peso)**  
  → `weight_condition_Obese` tem leve influência no aumento dos custos.  
  → `weight_condition_Overweight` e `weight_condition_Underweight` têm impacto menor comparado à obesidade.

---

### **Conclusões gerais:**
- Ser **fumante**, ter **idade elevada** e um **BMI alto** são os principais fatores que aumentam o custo médico previsto.
- Fatores como **sexo** e **região** têm influência, mas são significativamente menos relevantes.
## Modelo Final''')
plt.figure(figsize=(8,6))

# Gráfico de barras com médias
sns.barplot(x='children', y='charges', data=data, color='lightblue', ci=None)

# Linha de tendência (pointplot conecta as médias)
sns.pointplot(x='children', y='charges', data=data, color='red', markers='o', linestyles='--')

plt.title('Custos Médicos Médios por Número de Filhos')
plt.xlabel('Número de Filhos')
plt.ylabel('Custos Médicos Médios')

st.pyplot(plt.gcf())
plt.clf()
# https://www.cancer.org/cancer/risk-prevention/diet-physical-activity/body-weight-and-cancer-risk.html
# IMC entre 25,0 e 29,9 Kg/m2: sobrepeso; 
# IMC entre 30,0 e 34,9 Kg/m2: obesidade grau I; 
# IMC entre 35,0 e 39,9 Kg/m2: obesidade grau II; 
# IMC maior do que 40,0 Kg/m2: obesidade grau III.
data["weight_condition"] = ""
for col in [data]:
    col.loc[col["bmi"] < 18.5, "weight_condition"] = "Underweight"
    col.loc[(col["bmi"] >= 18.5) & (col["bmi"] < 24.986), "weight_condition"] = "Normal Weight"
    col.loc[(col["bmi"] >= 25) & (col["bmi"] < 29.926), "weight_condition"] = "Overweight"
    col.loc[col["bmi"] >= 30, "weight_condition"] = "Obese"
    

sns.lmplot(data=data, x="age", y="charges", hue='weight_condition', 
           scatter_kws = {"s": 10, "alpha": 0.3}) 
plt.title("Regressão Linear entre Idade e Custos Médicos por Fumantes e Obesidade")
plt.xlabel("Idade")
plt.ylabel("Custos Médicos ($)") 
st.pyplot(plt.gcf())
plt.clf()
print(data['smoker'].value_counts())
st.pyplot(plt.gcf())
plt.clf()
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_columns = ['sex', 'smoker', 'region']

# Criando um pre-processar para transformar as variáveis categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns) # categorical features
])

# Aplicar a transformação das categorias e nivelamento dos dados
base_transformed = preprocessor.fit_transform(data)
base_transformed_df = pd.DataFrame(base_transformed, columns=preprocessor.get_feature_names_out())
data_encoded = pd.concat([data.drop(columns=categorical_columns), base_transformed_df], axis=1)
data_encoded.head()
st.pyplot(plt.gcf())
plt.clf()
max_charge = data_encoded['charges'].max()
min_charge = data_encoded['charges'].min()

print(f"Maior valor de charges: {max_charge}")
print(f"Menor valor de charges: {min_charge}")

sns.boxplot(data_encoded['charges'])
plt.ylabel('Custos Médicos')
st.pyplot(plt.gcf())
plt.clf()
import numpy as np

Q1, Q3 = np.percentile(data_encoded['charges'], [25, 75])
print(f"Q1: {Q1}") # separa os 25% menores valores do conjunto de dados.
print(f"Q3: {Q3}") # separa os 25% maiores valores do conjunto de dados.

IQR = Q3 - Q1
lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)
print(f"Limite Inferior: {lower_bound}")
print(f"Limite Superior: {upper_bound}")

data_no_outliers = data_encoded[(data_encoded['charges'] >= lower_bound) & (data_encoded['charges'] <= upper_bound)]
data_no_outliers.head()
st.pyplot(plt.gcf())
plt.clf()
max_charge = data_no_outliers['charges'].max()
min_charge = data_no_outliers['charges'].min()

print(f"Maior valor de charges: {max_charge}")
print(f"Menor valor de charges: {min_charge}")

print("Original DataFrame Shape:", data.shape)
print("DataFrame Shape after Removing Outliers:", data_no_outliers.shape)

sns.boxplot(data_no_outliers['charges'])
plt.ylabel('Custos Médicos')
st.pyplot(plt.gcf())
plt.clf()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Copiar o dataset original para trabalhar sem risco
data_no_outliers = data.copy()

# Lista das colunas para analisar
columns = ['bmi', 'age']

# Criar subplots para boxplots antes da remoção dos outliers
fig, axs = plt.subplots(1, len(columns), figsize=(12, 5))

for i, column in enumerate(columns):
    sns.boxplot(y=data_no_outliers[column], ax=axs[i])
    axs[i].set_title(f'Boxplot de {column} (Antes)')
    axs[i].set_ylabel(column)

plt.tight_layout()

# Remover outliers usando IQR para cada coluna
for column in columns:
    Q1 = data_no_outliers[column].quantile(0.25)
    Q3 = data_no_outliers[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data_no_outliers = data_no_outliers[
        (data_no_outliers[column] >= lower_bound) & (data_no_outliers[column] <= upper_bound)
    ]

# Criar subplots para boxplots depois da remoção dos outliers
fig, axs = plt.subplots(1, len(columns), figsize=(12, 5))

for i, column in enumerate(columns):
    sns.boxplot(y=data_no_outliers[column], ax=axs[i])
    axs[i].set_title(f'Boxplot de {column} (Depois)')
    axs[i].set_ylabel(column)

plt.tight_layout()

# Mostrar resultado da remoção
print("Formato original do DataFrame:", data.shape)
print("Formato após remoção dos outliers:", data_no_outliers.shape)

data_no_outliers.head()
st.pyplot(plt.gcf())
plt.clf()
from scipy.stats import zscore

data_encoded['z_score'] = zscore(data_encoded['charges']) # Calcular o Z-Score para a coluna 'charges'

# Filtrar os dados para manter apenas os valores com Z-Score entre -3 e 3
data_no_outliers_z = data_encoded[(data_encoded['z_score'] > -3) & (data_encoded['z_score'] < 3)]
data_no_outliers_z = data_no_outliers_z.drop(columns=['z_score'])
data_no_outliers_z.head()
st.pyplot(plt.gcf())
plt.clf()
max_charge = data_no_outliers_z['charges'].max()
min_charge = data_no_outliers_z['charges'].min()

print(f"Maior valor de charges: {max_charge}")
print(f"Menor valor de charges: {min_charge}")

print("Original DataFrame Shape:", data.shape)
print("DataFrame Shape after Removing Outliers:", data_no_outliers_z.shape)

sns.boxplot(data_no_outliers_z['charges'])
plt.ylabel('Custos Médicos')
st.pyplot(plt.gcf())
plt.clf()
def detect_outliers_iqr(data, column):
    """Detecta outliers usando o método IQR"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(data, column, threshold=3):
    """Detecta outliers usando Z-score"""
    z_scores = np.abs(zscore(data[column]))
    outliers = data[z_scores > threshold]
    return outliers

# Análise de outliers para variáveis numéricas
numeric_cols = ['age', 'bmi', 'children', 'charges']
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

print("ANÁLISE DE OUTLIERS")

for i, col in enumerate(numeric_cols):
    # Boxplot original
    axes[0, i].boxplot(data[col])
    axes[0, i].set_title(f'{col.title()} - Original')
    axes[0, i].set_ylabel('Valores')
    
    # Detecção de outliers
    outliers_iqr, lower, upper = detect_outliers_iqr(data, col)
    outliers_z = detect_outliers_zscore(data, col)
    
    print(f"\n{col.upper()}:")
    print(f"• Outliers (IQR): {len(outliers_iqr)} ({len(outliers_iqr)/len(data)*100:.1f}%)")
    print(f"• Outliers (Z-score): {len(outliers_z)} ({len(outliers_z)/len(data)*100:.1f}%)")
    print(f"• Limites IQR: [{lower:.2f}, {upper:.2f}]")
    
    # Dados sem outliers (IQR)
    data_no_outliers = data[(data[col] >= lower) & (data[col] <= upper)]
    axes[1, i].boxplot(data_no_outliers[col])
    axes[1, i].set_title(f'{col.title()} - Sem Outliers')
    axes[1, i].set_ylabel('Valores')

plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()
print(data_no_outliers.columns)
st.pyplot(plt.gcf())
plt.clf()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Lista de colunas categóricas
categorical_columns = ['sex', 'smoker', 'region', 'weight_condition']

for col in categorical_columns:
    if col in data_no_outliers.columns:
        data_no_outliers[col] = label_encoder.fit_transform(data_no_outliers[col])


categorical_columns = [ 'weight_condition']
for col in  categorical_columns:
    data_no_outliers[col] = label_encoder.fit_transform(data_no_outliers[col])

# Matriz de correlação
correlation_matrix = data_no_outliers.corr()

# Visualizar a matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Greens', fmt='.2f')
plt.title('Matriz de Correlação')
st.pyplot(plt.gcf())
plt.clf()
# Criar uma cópia dos dados para processamento
data_processed = data.copy()

# One-Hot Encoding para variáveis categóricas
categorical_columns = ['sex', 'smoker', 'region']
data_encoded = pd.get_dummies(data_processed, columns=categorical_columns, drop_first=True)

# Remoção de outliers baseada em charges (variável target)
outliers_charges, lower_bound, upper_bound = detect_outliers_iqr(data_encoded, 'charges')
data_final = data_encoded[(data_encoded['charges'] >= lower_bound) & 
                         (data_encoded['charges'] <= upper_bound)]



# Matriz de correlação final
plt.figure(figsize=(14, 10))
correlation_matrix = data_final.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
            center=0, square=True, linewidths=0.5)
plt.title('Matriz de Correlação Final (Dados Processados)', fontsize=14, fontweight='bold')
plt.tight_layout()

# Correlações com a variável target
target_correlations = correlation_matrix['charges'].abs().sort_values(ascending=False)
print("\n CORRELAÇÕES COM CUSTOS MÉDICOS (em ordem decrescente): \n")

for var, corr in target_correlations.items():
    if var != 'charges':
        strength = "Forte" if corr >= 0.7 else "Moderada" if corr >= 0.3 else "Fraca"
        print(f"• {var:20} : {corr:.3f} ({strength})")
st.pyplot(plt.gcf())
plt.clf()
print("PRINCIPAIS INSIGHTS DA ANÁLISE\n")

# Insight 1: Impacto do tabagismo
smoker_impact = data.groupby('smoker')['charges'].mean()
impact_ratio = smoker_impact['yes'] / smoker_impact['no']
print(f"1. TABAGISMO:")
print(f"   • Fumantes pagam {impact_ratio:.1f}x mais que não fumantes")
print(f"   • Diferença média: ${smoker_impact['yes'] - smoker_impact['no']:.0f}")

# Insight 2: Impacto da idade
age_correlation = data['age'].corr(data['charges'])
print(f"\n2. IDADE:")
print(f"   • Correlação com custos: {age_correlation:.3f}")
print(f"   • A cada ano adicional: ~${(data['charges'].max() - data['charges'].min()) / (data['age'].max() - data['age'].min()):.0f} de aumento")

# Insight 3: Impacto do IMC
bmi_obese = data[data['bmi'] == 'Obesidade']['charges'].mean()
bmi_normal = data[data['bmi'] == 'Peso normal']['charges'].mean()
print(f"\n3. IMC:")
print(f"   • Pessoas obesas pagam ${bmi_obese - bmi_normal:.0f} a mais que peso normal")
print(f"   • Aumento percentual: {((bmi_obese / bmi_normal) - 1) * 100:.1f}%")

# Insight 4: Combinação de fatores
high_risk = data[(data['smoker'] == 'yes') & (data['bmi'] >= 30)]
low_risk = data[(data['smoker'] == 'no') & (data['bmi'] < 25)]

if len(high_risk) > 0 and len(low_risk) > 0:
    print(f"\n4. PERFIL DE RISCO:")
    print(f"   • Alto risco (fumante + obeso): ${high_risk['charges'].mean():.0f}")
    print(f"   • Baixo risco (não fumante + peso normal): ${low_risk['charges'].mean():.0f}")
    print(f"   • Diferença: {high_risk['charges'].mean() / low_risk['charges'].mean():.1f}x")
st.pyplot(plt.gcf())
plt.clf()
# Separação de features e target
feature_columns = [col for col in data_final.columns if col != 'charges']
X = data_final[feature_columns]
y = data_final['charges']

print("DADOS PREPARADOS PARA MODELAGEM:\n")
print(f"• Features (X): {X.shape}")
print(f"• Target (y): {y.shape}")
print(f"• Variáveis preditoras: {list(X.columns)}")

# Estatísticas finais
print(f"\nESTATÍSTICAS FINAIS DO TARGET:\n")
print(f"• Média: ${y.mean():.2f}")
print(f"• Mediana: ${y.median():.2f}")
print(f"• Desvio padrão: ${y.std():.2f}")
print(f"• Coeficiente de variação: {(y.std()/y.mean())*100:.1f}%")

# Salvamento dos dados processados (opcional)
# data_final.to_csv('dados_processados_insurance.csv', index=False)
print("\nAnálise exploratória concluída!")
print("Dados prontos para modelagem preditiva")
st.pyplot(plt.gcf())
plt.clf()
print(data['smoker'].value_counts())
st.pyplot(plt.gcf())
plt.clf()
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_columns = ['sex', 'smoker', 'region']

# Criando um pre-processar para transformar as variáveis categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns) # categorical features
])

# Aplicar a transformação das categorias e nivelamento dos dados
base_transformed = preprocessor.fit_transform(data)
base_transformed_df = pd.DataFrame(base_transformed, columns=preprocessor.get_feature_names_out())
data_encoded = pd.concat([data.drop(columns=categorical_columns), base_transformed_df], axis=1)
data_encoded.head()
st.pyplot(plt.gcf())
plt.clf()
max_charge = data_encoded['charges'].max()
min_charge = data_encoded['charges'].min()

print(f"Maior valor de charges: {max_charge}")
print(f"Menor valor de charges: {min_charge}")

sns.boxplot(data_encoded['charges'])
plt.ylabel('Custos Médicos')
st.pyplot(plt.gcf())
plt.clf()
import numpy as np

Q1, Q3 = np.percentile(data_encoded['charges'], [25, 75])
print(f"Q1: {Q1}") # separa os 25% menores valores do conjunto de dados.
print(f"Q3: {Q3}") # separa os 25% maiores valores do conjunto de dados.

IQR = Q3 - Q1
lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)
print(f"Limite Inferior: {lower_bound}")
print(f"Limite Superior: {upper_bound}")

data_no_outliers = data_encoded[(data_encoded['charges'] >= lower_bound) & (data_encoded['charges'] <= upper_bound)]
data_no_outliers.head()
st.pyplot(plt.gcf())
plt.clf()
max_charge = data_no_outliers['charges'].max()
min_charge = data_no_outliers['charges'].min()

print(f"Maior valor de charges: {max_charge}")
print(f"Menor valor de charges: {min_charge}")

print("Original DataFrame Shape:", data.shape)
print("DataFrame Shape after Removing Outliers:", data_no_outliers.shape)

sns.boxplot(data_no_outliers['charges'])
plt.ylabel('Custos Médicos')
st.pyplot(plt.gcf())
plt.clf()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Copiar o dataset original para trabalhar sem risco
data_no_outliers = data.copy()

# Lista das colunas para analisar
columns = ['bmi', 'age']

# Criar subplots para boxplots antes da remoção dos outliers
fig, axs = plt.subplots(1, len(columns), figsize=(12, 5))

for i, column in enumerate(columns):
    sns.boxplot(y=data_no_outliers[column], ax=axs[i])
    axs[i].set_title(f'Boxplot de {column} (Antes)')
    axs[i].set_ylabel(column)

plt.tight_layout()

# Remover outliers usando IQR para cada coluna
for column in columns:
    Q1 = data_no_outliers[column].quantile(0.25)
    Q3 = data_no_outliers[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data_no_outliers = data_no_outliers[
        (data_no_outliers[column] >= lower_bound) & (data_no_outliers[column] <= upper_bound)
    ]

# Criar subplots para boxplots depois da remoção dos outliers
fig, axs = plt.subplots(1, len(columns), figsize=(12, 5))

for i, column in enumerate(columns):
    sns.boxplot(y=data_no_outliers[column], ax=axs[i])
    axs[i].set_title(f'Boxplot de {column} (Depois)')
    axs[i].set_ylabel(column)

plt.tight_layout()

# Mostrar resultado da remoção
print("Formato original do DataFrame:", data.shape)
print("Formato após remoção dos outliers:", data_no_outliers.shape)

data_no_outliers.head()
st.pyplot(plt.gcf())
plt.clf()
from scipy.stats import zscore

data_encoded['z_score'] = zscore(data_encoded['charges']) # Calcular o Z-Score para a coluna 'charges'

# Filtrar os dados para manter apenas os valores com Z-Score entre -3 e 3
data_no_outliers_z = data_encoded[(data_encoded['z_score'] > -3) & (data_encoded['z_score'] < 3)]
data_no_outliers_z = data_no_outliers_z.drop(columns=['z_score'])
data_no_outliers_z.head()
st.pyplot(plt.gcf())
plt.clf()
max_charge = data_no_outliers_z['charges'].max()
min_charge = data_no_outliers_z['charges'].min()

print(f"Maior valor de charges: {max_charge}")
print(f"Menor valor de charges: {min_charge}")

print("Original DataFrame Shape:", data.shape)
print("DataFrame Shape after Removing Outliers:", data_no_outliers_z.shape)

sns.boxplot(data_no_outliers_z['charges'])
plt.ylabel('Custos Médicos')
st.pyplot(plt.gcf())
plt.clf()
def detect_outliers_iqr(data, column):
    """Detecta outliers usando o método IQR"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(data, column, threshold=3):
    """Detecta outliers usando Z-score"""
    z_scores = np.abs(zscore(data[column]))
    outliers = data[z_scores > threshold]
    return outliers

# Análise de outliers para variáveis numéricas
numeric_cols = ['age', 'bmi', 'children', 'charges']
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

print("ANÁLISE DE OUTLIERS")

for i, col in enumerate(numeric_cols):
    # Boxplot original
    axes[0, i].boxplot(data[col])
    axes[0, i].set_title(f'{col.title()} - Original')
    axes[0, i].set_ylabel('Valores')
    
    # Detecção de outliers
    outliers_iqr, lower, upper = detect_outliers_iqr(data, col)
    outliers_z = detect_outliers_zscore(data, col)
    
    print(f"\n{col.upper()}:")
    print(f"• Outliers (IQR): {len(outliers_iqr)} ({len(outliers_iqr)/len(data)*100:.1f}%)")
    print(f"• Outliers (Z-score): {len(outliers_z)} ({len(outliers_z)/len(data)*100:.1f}%)")
    print(f"• Limites IQR: [{lower:.2f}, {upper:.2f}]")
    
    # Dados sem outliers (IQR)
    data_no_outliers = data[(data[col] >= lower) & (data[col] <= upper)]
    axes[1, i].boxplot(data_no_outliers[col])
    axes[1, i].set_title(f'{col.title()} - Sem Outliers')
    axes[1, i].set_ylabel('Valores')

plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()
print(data_no_outliers.columns)
st.pyplot(plt.gcf())
plt.clf()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Lista de colunas categóricas
categorical_columns = ['sex', 'smoker', 'region', 'weight_condition']

for col in categorical_columns:
    if col in data_no_outliers.columns:
        data_no_outliers[col] = label_encoder.fit_transform(data_no_outliers[col])


categorical_columns = [ 'weight_condition']
for col in  categorical_columns:
    data_no_outliers[col] = label_encoder.fit_transform(data_no_outliers[col])

# Matriz de correlação
correlation_matrix = data_no_outliers.corr()

# Visualizar a matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Greens', fmt='.2f')
plt.title('Matriz de Correlação')
st.pyplot(plt.gcf())
plt.clf()
data = pd.read_csv("dados/insurance.csv")
data = data.drop(['region'], axis='columns')
#data = data.drop(['sex'], axis='columns')
data.head()
st.pyplot(plt.gcf())
plt.clf()
# Criar uma cópia dos dados para processamento
data_processed = data.copy()

# One-Hot Encoding para variáveis categóricas
categorical_columns = ['sex', 'smoker', 'region', 'weight_condition']
data_encoded = pd.get_dummies(data_processed, columns=categorical_columns, drop_first=True)

# Remoção de outliers baseada em charges (variável target)
outliers_charges, lower_bound, upper_bound = detect_outliers_iqr(data_encoded, 'charges')
data_final = data_encoded[(data_encoded['charges'] >= lower_bound) & 
                         (data_encoded['charges'] <= upper_bound)]



# Matriz de correlação final
plt.figure(figsize=(14, 10))
correlation_matrix = data_final.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
            center=0, square=True, linewidths=0.5)
plt.title('Matriz de Correlação Final (Dados Processados)', fontsize=14, fontweight='bold')
plt.tight_layout()

# Correlações com a variável target
target_correlations = correlation_matrix['charges'].abs().sort_values(ascending=False)
print("\n CORRELAÇÕES COM CUSTOS MÉDICOS (em ordem decrescente): \n")

for var, corr in target_correlations.items():
    if var != 'charges':
        strength = "Forte" if corr >= 0.7 else "Moderada" if corr >= 0.3 else "Fraca"
        print(f"• {var:20} : {corr:.3f} ({strength})")
st.pyplot(plt.gcf())
plt.clf()
print("PRINCIPAIS INSIGHTS DA ANÁLISE\n")

# Insight 1: Impacto do tabagismo
smoker_impact = data.groupby('smoker')['charges'].mean()
impact_ratio = smoker_impact['yes'] / smoker_impact['no']
print(f"1. TABAGISMO:")
print(f"   • Fumantes pagam {impact_ratio:.1f}x mais que não fumantes")
print(f"   • Diferença média: ${smoker_impact['yes'] - smoker_impact['no']:.0f}")

# Insight 2: Impacto da idade
age_correlation = data['age'].corr(data['charges'])
print(f"\n2. IDADE:")
print(f"   • Correlação com custos: {age_correlation:.3f}")
print(f"   • A cada ano adicional: ~${(data['charges'].max() - data['charges'].min()) / (data['age'].max() - data['age'].min()):.0f} de aumento")

# Insight 3: Impacto do IMC
bmi_obese = data[data['bmi'] == 'Obesidade']['charges'].mean()
bmi_normal = data[data['bmi'] == 'Peso normal']['charges'].mean()
print(f"\n3. IMC:")
print(f"   • Pessoas obesas pagam ${bmi_obese - bmi_normal:.0f} a mais que peso normal")
print(f"   • Aumento percentual: {((bmi_obese / bmi_normal) - 1) * 100:.1f}%")

# Insight 4: Combinação de fatores
high_risk = data[(data['smoker'] == 'yes') & (data['bmi'] >= 30)]
low_risk = data[(data['smoker'] == 'no') & (data['bmi'] < 25)]

if len(high_risk) > 0 and len(low_risk) > 0:
    print(f"\n4. PERFIL DE RISCO:")
    print(f"   • Alto risco (fumante + obeso): ${high_risk['charges'].mean():.0f}")
    print(f"   • Baixo risco (não fumante + peso normal): ${low_risk['charges'].mean():.0f}")
    print(f"   • Diferença: {high_risk['charges'].mean() / low_risk['charges'].mean():.1f}x")
st.pyplot(plt.gcf())
plt.clf()
# Criando a coluna 'weight_condition'
conditions = [
    (data['bmi'] < 18.5),
    (data['bmi'] >= 18.5) & (data['bmi'] < 24.986),
    (data['bmi'] >= 25) & (data['bmi'] < 29.926),
    (data['bmi'] >= 30)
]

labels = ['Abaixo do Peso', 'Peso Normal', 'Sobrepeso', 'Obesidade']

data['weight_condition'] = np.select(conditions, labels, default='Unknown')

# Plotando o gráfico
plt.figure(figsize=(6,4))
sns.countplot(x='weight_condition', data=data, palette='Set2', order=['Abaixo do Peso', 'Peso Normal', 'Sobrepeso', 'Obesidade'])
plt.title('Distribuição por Condição de Peso (BMI)', fontsize=14)
plt.xlabel('Condição de Peso')
plt.ylabel('Quantidade de Pessoas')
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Preparando os dados
numeric_columns = [ 'age', 'bmi', 'children'] # Colunas numéricas
categorical_columns = [ 'smoker', 'weight_condition']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns), # numeric features
        ('cat', OneHotEncoder(), categorical_columns) # categorical features
    ])


base_transformed = preprocessor.fit_transform(data)
base_transformed
st.pyplot(plt.gcf())
plt.clf()
base_transformed_df = pd.DataFrame(base_transformed, columns=preprocessor.get_feature_names_out())
data = pd.concat([data['charges'], base_transformed_df], axis=1)
data.head()
st.pyplot(plt.gcf())
plt.clf()
# Matriz de correlação
correlation_matrix = data.corr()
print(correlation_matrix["charges"].sort_values(ascending=False))
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Greens', fmt='.2f')
plt.title('Matriz de Correlação')
st.pyplot(plt.gcf())
plt.clf()
# Separação de features e target
feature_columns = [col for col in data_final.columns if col != 'charges']
X = data_final[feature_columns]
y = data_final['charges']

print("DADOS PREPARADOS PARA MODELAGEM:\n")
print(f"• Features (X): {X.shape}")
print(f"• Target (y): {y.shape}")
print(f"• Variáveis preditoras: {list(X.columns)}")

# Estatísticas finais
print(f"\nESTATÍSTICAS FINAIS DO TARGET:\n")
print(f"• Média: ${y.mean():.2f}")
print(f"• Mediana: ${y.median():.2f}")
print(f"• Desvio padrão: ${y.std():.2f}")
print(f"• Coeficiente de variação: {(y.std()/y.mean())*100:.1f}%")

# Salvamento dos dados processados (opcional)
# data_final.to_csv('dados_processados_insurance.csv', index=False)
print("\nAnálise exploratória concluída!")
print("Dados prontos para modelagem preditiva")
st.pyplot(plt.gcf())
plt.clf()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
st.pyplot(plt.gcf())
plt.clf()
data_encoded = pd.get_dummies(data, columns=['sex', 'smoker', 'region', 'weight_condition'], drop_first=True)

X = data_encoded.drop(['charges'], axis=1)
y = data_encoded['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
st.pyplot(plt.gcf())
plt.clf()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")

def calculate_mape(labels, predictions):
    errors = np.abs(labels - predictions)
    relative_errors = errors / np.abs(labels)
    mape = np.mean(relative_errors) * 100
    return mape

def evaluate_model(model, y_test, y_pred):
    print(f"\n{'='*100}")
    print(f"Avaliação do Modelo: {model}")
    print(f"{'='*100}")
    print("* Erro Absoluto Médio (MAE):", round(mean_absolute_error(y_test, y_pred), 2))
    print("* Erro Quadrático Médio (MSE):", round(mean_squared_error(y_test, y_pred), 2))
    print("* Raiz do Erro Quadrático Médio (RMSE):", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
    print("* MAPE:", round(calculate_mape(y_test, y_pred), 2), "%")
    print("* R²:", round(r2_score(y_test, y_pred), 4))

def plot_residuals(model, y_test, y_pred):
    residuals = y_test - y_pred

    # Gráfico de Resíduos
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, color='royalblue', alpha=0.6, s=60)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.title(f'Resíduos vs Previsões | {model.__class__.__name__}', fontsize=14, weight='bold')
    plt.xlabel('Previsões', fontsize=12)
    plt.ylabel('Resíduos', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Gráfico de Dispersão (Valores Reais vs Previsto)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='seagreen', alpha=0.6, s=60)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.title(f'Valores Reais vs Valores Previsto | {model.__class__.__name__}', fontsize=14, weight='bold')
    plt.xlabel('Valores Reais', fontsize=12)
    plt.ylabel('Valores Previsto', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

def predict_and_evaluate(model, X_test, X_train, y_test, y_train):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plot_residuals(model, y_test, y_pred)
    evaluate_model(model, y_test, y_pred)
st.pyplot(plt.gcf())
plt.clf()
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

models = [
    LinearRegression(), 
    DecisionTreeRegressor(random_state=42), 
    RandomForestRegressor(n_estimators=100, random_state=42),
    KNeighborsRegressor(n_neighbors=5)
]

for model in models:
    predict_and_evaluate(model, X_test, X_train, y_test, y_train)
st.pyplot(plt.gcf())
plt.clf()
model = RandomForestRegressor(n_estimators=7, max_depth=20, max_leaf_nodes=17, criterion = 'friedman_mse', random_state = 1)
predict_and_evaluate(model, X_test, X_train, y_test, y_train)
st.pyplot(plt.gcf())
plt.clf()
import shap

# Cria o explainer com uma função que faz a predição
explainer = shap.KernelExplainer(model.predict, X_train)

# Calcula os valores SHAP para um subconjunto dos dados
shap_values = explainer.shap_values(X_test[:50])

# Plot do impacto das variáveis
shap.summary_plot(shap_values, X_test[:50])
st.pyplot(plt.gcf())
plt.clf()
import pickle
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class WeightConditionTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["weight_condition"] = ""
        X.loc[X["bmi"] < 18.5, "weight_condition"] = "Underweight"
        X.loc[(X["bmi"] >= 18.5) & (X["bmi"] < 25), "weight_condition"] = "Normal Weight"
        X.loc[(X["bmi"] >= 25) & (X["bmi"] < 30), "weight_condition"] = "Overweight"
        X.loc[X["bmi"] >= 30, "weight_condition"] = "Obese"
        return X

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['age', 'bmi', 'children']), # numeric features
    ('cat', OneHotEncoder(), ['smoker', "weight_condition"]) # categorical features
])

pipeline = Pipeline(steps=[
    ('add_weight_condition', WeightConditionTransformer()), # Adiciona a coluna weight_condition
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=7, max_depth=20, max_leaf_nodes=17, criterion = 'friedman_mse', random_state = 1))
])

data = pd.read_csv("dados/insurance.csv")
data = data.drop(['region'], axis='columns')  # A Coluna Região não é relevante para o modelo
data = data.drop(['sex'], axis='columns')  # A Coluna Sexo não é relevante para o modelo


X = data.drop(['charges'], axis=1)
y = data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
pipeline.fit(X_train, y_train)

with open('wcmodel.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
    
pipeline
st.pyplot(plt.gcf())
plt.clf()
features = pd.DataFrame([['no', 33,25, 0]], columns=['smoker', 'age', 'bmi', 'children'])
result = pipeline.predict(features)
print(f"Predicted insurance cost: {result}")
st.pyplot(plt.gcf())
plt.clf()
import os
from pydantic import BaseModel


class InsuranceData(BaseModel):
    smoker: str
    age: int
    bmi: float
    children: int
    

class WealthCostPrediction:
    
    def __init__(self):
        self.model_path = 'wcmodel.pkl'
        self.pipeline = None
        
    def predict(self, features):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file '{self.model_path}' not found. Please train the model first.")
        
        with open(self.model_path, 'rb') as f:
            self.pipeline = pickle.load(f)
        
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame([features], columns=['smoker', 'age', 'bmi', 'children'])
        
        prediction = self.pipeline.predict(features)
        return round(float(prediction[0]), 2)
        


features = InsuranceData(smoker='no', age=33, bmi=25, children=0)


prediction = WealthCostPrediction()
result = prediction.predict([features.smoker, features.age, features.bmi, features.children])
print(f"Predicted insurance cost: {result}")
st.pyplot(plt.gcf())
plt.clf()