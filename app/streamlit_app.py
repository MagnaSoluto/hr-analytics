import streamlit as st
import requests

API_URL = "http://localhost:8000/prever"


def calcular_faixa_etaria(idade: int) -> str:
    if idade < 25:
        return "18-24"
    if idade < 35:
        return "25-34"
    if idade < 45:
        return "35-44"
    if idade < 55:
        return "45-54"
    if idade < 65:
        return "55-64"
    return "65+"

st.set_page_config(
    page_title="HR Turnover Predictor",
    page_icon="🏢",
    layout="centered",
)

st.title("Previsão de Desligamento")


with st.form("funcionario_form"):
    st.header("Informações Pessoais")
    col1, col2, col3 = st.columns(3)
    with col1:
        idade = st.number_input("Idade", min_value=18, max_value=70, value=30)
        faixa_etaria = calcular_faixa_etaria(idade)
        st.caption(f"Faixa Etária: {faixa_etaria}")
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
    faixa_etaria = calcular_faixa_etaria(idade)
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
