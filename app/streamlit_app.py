import streamlit as st
import requests

API_URL = "http://localhost:8000/prever"

st.set_page_config(
    page_title="Mackenzie Turnover Predictor",
    page_icon="🏢",
    layout="centered",
)

# Estilo visual para título, subheadings e labels
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
    }

    /* Título principal */
    .title {
        color: #A80532;
        text-align: center;
        font-size: 36px;
        font-weight: 700;
    }

    /* Subtítulos (como Informações Pessoais, Carreira etc.) */
    section div[data-testid="stMarkdownContainer"] h2 {
        color: #1f1f1f;
        font-size: 20px;
        font-weight: 700;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }

    /* Labels dos campos */
    label, .stRadio > label {
        font-weight: 600 !important;
        color: #1f1f1f !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "# <span class='title'>Previsão de Desligamento</span>", unsafe_allow_html=True
)

# Estilos adicionais para melhorar visibilidade dos rótulos
st.markdown(
    """
    <style>
    label, .stRadio > label {
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.form("funcionario_form"):
    st.markdown('<div class="secao-titulo">Informações Pessoais</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        idade = st.number_input("Idade", min_value=18, max_value=70, value=30)
    with col2:
        genero = st.selectbox("Gênero", ["Masculino", "Feminino", "Outro"])
    with col3:
        estado_civil = st.selectbox(
            "Estado Civil", ["Solteiro", "Casado", "Divorciado"]
        )

    st.subheader("Carreira")
    col1, col2 = st.columns(2)
    with col1:
        nivel_cargo = st.selectbox(
            "Nível do Cargo",
            ["Júnior", "Pleno", "Sênior", "Gerente", "Diretor"],
        )
        cargo = st.selectbox(
            "Cargo",
            [
                "Analista",
                "Desenvolvedor",
                "Gerente",
                "Coordenador",
                "Diretor",
                "Estagiário",
            ],
        )
        escolaridade = st.selectbox(
            "Escolaridade",
            [
                "Ensino Médio",
                "Tecnólogo",
                "Superior",
                "Pós-graduação",
                "Mestrado",
                "Doutorado",
            ],
        )
    with col2:
        setor = st.selectbox(
            "Setor",
            ["TI", "RH", "Financeiro", "Marketing", "Vendas", "Operações"],
        )
        faixa_etaria = st.selectbox(
            "Faixa Etária",
            ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        )
        salario_mensal = st.number_input("Salário Mensal", min_value=0.0, value=5000.0)
        anos_experiencia = st.number_input(
            "Anos de Experiência", min_value=0.0, value=5.0
        )
        anos_empresa = st.number_input("Anos na Empresa", min_value=0.0, value=3.0)

    col3, col4 = st.columns(2)
    with col3:
        anos_cargo_atual = st.number_input(
            "Anos no Cargo Atual", min_value=0.0, value=2.0
        )
        anos_ultima_promocao = st.number_input(
            "Anos desde a Última Promoção", min_value=0.0, value=1.0
        )
        tempo_sem_promocao = st.number_input(
            "Tempo sem Promoção", min_value=0.0, value=1.0
        )
        viagem_trabalho = st.selectbox(
            "Viagem a Trabalho",
            ["Não viaja", "Às vezes", "Frequentemente"],
        )
        horas_extras = st.number_input("Horas Extras", min_value=0.0, value=5.0)
    with col4:
        area_formacao = st.selectbox(
            "Área de Formação",
            ["TI", "Engenharia", "Administração", "RH", "Outros"],
        )
        satisfacao_trabalho = st.slider("Satisfação no Trabalho", 1, 5, 3)
        satisfacao_ambiente = st.slider("Satisfação com o Ambiente", 1, 5, 3)
        satisfacao_relacionamento = st.slider("Satisfação com Relacionamentos", 1, 5, 3)
        equilibrio_vida = st.slider("Equilíbrio Vida-Trabalho", 1, 5, 3)

    col5, col6 = st.columns(2)
    with col5:
        empresas_anteriores = st.number_input(
            "Empresas Anteriores", min_value=0, value=1
        )
        anos_gestor_atual = st.number_input(
            "Anos com o Gestor Atual", min_value=0.0, value=1.0
        )
        distancia_casa = st.number_input(
            "Distância de Casa (km)", min_value=0.0, value=10.0
        )

    submitted = st.form_submit_button("Prever")

if submitted:
    payload = {
        "funcionarios": [
            {
                "Idade": idade,
                "Genero": genero,
                "EstadoCivil": estado_civil,
                "NivelCargo": nivel_cargo,
                "Cargo": cargo,
                "Escolaridade": escolaridade,
                "Setor": setor,
                "FaixaEtaria": faixa_etaria,
                "SalarioMensal": salario_mensal,
                "AnosExperiencia": anos_experiencia,
                "AnosEmpresa": anos_empresa,
                "AnosCargoAtual": anos_cargo_atual,
                "AnosUltimaPromocao": anos_ultima_promocao,
                "TempoSemPromocao": tempo_sem_promocao,
                "ViagemTrabalho": viagem_trabalho,
                "HorasExtras": horas_extras,
                "AreaFormacao": area_formacao,
                "SatisfacaoTrabalho": satisfacao_trabalho,
                "SatisfacaoAmbiente": satisfacao_ambiente,
                "SatisfacaoRelacionamento": satisfacao_relacionamento,
                "EquilibrioVida": equilibrio_vida,
                "EmpresasAnteriores": empresas_anteriores,
                "AnosGestorAtual": anos_gestor_atual,
                "DistanciaCasa": distancia_casa,
            }
        ]
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            prob = result["resultados"][0]["probabilidade_desligamento"]
            st.success(f"Probabilidade de desligamento: {prob:.2%}")
        else:
            st.error("Erro ao chamar a API")
    except Exception as e:
        st.error(f"Erro: {e}")
