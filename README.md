
# HR Analytics - Previsão de Rotatividade de Funcionários

Projeto desenvolvido como Trabalho Final da disciplina **Machine Learning Aplicado: HR Analytics Challenge** do MBA.

---

## 📘 Contexto do Problema

A TechCorp Brasil, uma das maiores empresas de tecnologia do país, enfrentou um aumento de **35% na rotatividade de funcionários** no último ano, representando um custo estimado de **R$ 45 milhões**.

Cada desligamento representa:
- Perda de conhecimento institucional  
- Queda na produtividade  
- Impacto na moral da equipe  
- Custos de desligamento e reposição (1.5x o salário anual)

---

## 🎯 Objetivo

Construir um **pipeline completo de Machine Learning** que:
- Faça a análise exploratória de dados (EDA)
- Realize o tratamento e a criação de variáveis (feature engineering)
- Modele e avalie preditores para o risco de desligamento
- Lide adequadamente com o desbalanceamento da variável alvo
- Comunique os resultados por meio de visualizações, relatórios e aplicação interativa

---

## 🧠 Sobre o Dataset

Os dados utilizados foram gerados sinteticamente com base em características do mercado de trabalho brasileiro.  
O dataset simula 1 milhão de registros com as seguintes dimensões:

- **Demográficas**: idade, gênero, escolaridade, estado civil  
- **Profissionais**: setor, cargo, anos de experiência, anos na empresa  
- **Satisfação**: ambiente, trabalho, relacionamento, equilíbrio vida-trabalho  
- **Desempenho**: nível do cargo, salário, histórico de promoções  
- **Variável alvo**: `Desligamento` (Sim/Não)

---

## ⚙️ Como Executar o Projeto

### 1. Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/hr-analytics.git
cd hr-analytics
```

### 2. Criar e Ativar o Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows
```

### 3. Instalar as Dependências

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Executar as Etapas do Projeto

Os notebooks devem ser executados na seguinte ordem dentro da pasta `notebooks/`:

- 01_eda.ipynb
- 02_feature_engineering.ipynb
- 03_modelagem.ipynb
- 04_optuna_tuning.ipynb
- 05_interpretabilidade.ipynb

**Para geração do dataset sintético:**  
O script `scripts/gerar_dataset.py` permite gerar novas amostras controladas via função `gerar_base_rh_analitico`.

**Para executar a API:**
Após salvar o modelo treinado na etapa 5, inicialize o serviço FastAPI utilizando o `uvicorn`:
```bash
uvicorn app.main:app --reload
```

**Para iniciar a interface Streamlit:**
Em outro terminal, execute o aplicativo que consome a API:
```bash
streamlit run app/streamlit_app.py
```

---

## 📁 Estrutura de Pastas

```
hr-analytics/
├── app/                    # Código da aplicação Streamlit ou API
├── data/                   # Dataset bruto e processado
├── models/                 # Modelos treinados (.pkl)
├── notebooks/              # Notebooks Jupyter por etapa
├── reports/                # Relatórios e gráficos gerados
├── scripts/                # Scripts utilitários
├── requirements.txt        # Dependências do projeto
└── README.md               # Este documento
```

---

## ✅ Tecnologias Utilizadas

- Python 3.11+
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn
- Imbalanced-learn (SMOTE)
- LightGBM
- SHAP
- Optuna
- Streamlit / FastAPI

---

## 📅 Informações da Entrega

- **Disciplina**: Machine Learning Aplicado — MBA
- **Professor**: Matheus H. P. Pacheco
- **Data limite**: 22/07/2025
- **Valor**: 10 pontos

---

## ✍️ Autor

- Adriano Carvalho dos Santos - 10747203  
- Mackenzie - MBA em Engenharia de Dados - Turma G 2025  
- GitHub: https://github.com/MagnaSoluto  
- LinkedIn: https://linkedin.com/in/Drico2236

---

> “Em Deus confiamos. Todos os outros tragam dados.” – W. Edwards Deming
