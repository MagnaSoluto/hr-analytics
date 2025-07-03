from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Literal
import pandas as pd
import joblib
import os

# ========== Inicialização ==========
app = FastAPI(
    title="API de Previsão de Desligamento",
    description="Retorna a probabilidade de desligamento de funcionários com base no modelo LightGBM treinado.",
    version="1.0"
)

# ========== Caminhos ==========
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(root_dir, 'models', 'lightgbm_model.pkl')
preproc_path = os.path.join(root_dir, 'models', 'preprocessor.pkl')

# ========== Carregamento ==========
modelo = joblib.load(model_path)
preprocessador = joblib.load(preproc_path)

# Função para preparar dados com feature engineering e codificação simples
def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features e codifica categorias para uso no modelo."""

    df = df.copy()

    # Novas features baseadas em regras do domínio
    df['SatisfacaoMedia'] = df[['SatisfacaoTrabalho', 'SatisfacaoAmbiente', 'SatisfacaoRelacionamento']].mean(axis=1)
    df['FaixaEtaria'] = pd.cut(df['Idade'], bins=[17, 29, 45, 66], labels=['<30', '30-45', '>45'])
    df['DistanteDoTrabalho'] = (df['DistanciaCasa'] > 20).astype(int)
    df['ViajaMuito'] = (df['ViagemTrabalho'] == 'Frequentemente').astype(int)
    df['FormacaoSuperiorOuMais'] = df['Escolaridade'].apply(
        lambda x: 1 if x in ['Superior', 'Pós-graduação', 'Mestrado', 'Doutorado'] else 0
    )
    df['RiscoHorasExtras'] = ((df['HorasExtras'] > 10) & (df['EquilibrioVida'] <= 2)).astype(int)
    df['TempoSemPromocao'] = df['AnosUltimaPromocao']
    df['ExperienciaPorEmpresa'] = df['AnosExperiencia'] / (df['EmpresasAnteriores'] + 1)

    nivel_map = {'Júnior': 1, 'Pleno': 2, 'Sênior': 3, 'Gerente': 4, 'Diretor': 5}
    df['SalarioAjustado'] = df['SalarioMensal'] / df['NivelCargo'].map(nivel_map).replace(0, 1)
    df['EstabilidadeNaEmpresa'] = df['AnosEmpresa'] / df['AnosExperiencia'].replace(0, 1)

    # Codificação simples (ordinal) para as colunas categóricas
    mapeamentos = {
        'Genero': {'Feminino': 0, 'Masculino': 1, 'Outro': 2},
        'EstadoCivil': {'Casado': 0, 'Divorciado': 1, 'Solteiro': 2},
        'NivelCargo': nivel_map,
        'Cargo': {
            k: i for i, k in enumerate(sorted(['Analista', 'Coordenador', 'Desenvolvedor', 'Diretor', 'Estagiário', 'Gerente']))
        },
        'Escolaridade': {
            'Ensino Médio': 1,
            'Tecnólogo': 2,
            'Superior': 3,
            'Pós-graduação': 4,
            'Mestrado': 5,
            'Doutorado': 6,
        },
        'Setor': {k: i for i, k in enumerate(sorted(['TI', 'RH', 'Financeiro', 'Marketing', 'Vendas', 'Operações']))},
        'ViagemTrabalho': {'Não viaja': 0, 'Às vezes': 1, 'Frequentemente': 2},
        'AreaFormacao': {k: i for i, k in enumerate(sorted(['TI', 'Engenharia', 'Administração', 'RH', 'Outros']))},
        'FaixaEtaria': {'<30': 0, '30-45': 1, '>45': 2},
    }

    for coluna, mapa in mapeamentos.items():
        if coluna in df.columns:
            df[coluna] = df[coluna].map(mapa)

    return df

# ========== Schema ==========
class Funcionario(BaseModel):
    Idade: int = Field(..., example=34)
    Genero: Literal["Masculino", "Feminino", "Outro"]
    EstadoCivil: Literal["Solteiro", "Casado", "Divorciado"]
    NivelCargo: Literal["Júnior", "Pleno", "Sênior", "Gerente", "Diretor"]
    Cargo: Literal["Analista", "Desenvolvedor", "Gerente", "Coordenador", "Diretor", "Estagiário"]
    Escolaridade: Literal["Ensino Médio", "Tecnólogo", "Superior", "Pós-graduação", "Mestrado", "Doutorado"]
    Setor: Literal["TI", "RH", "Financeiro", "Marketing", "Vendas", "Operações"]
    FaixaEtaria: Literal["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    SalarioMensal: float = Field(..., example=5500.0)
    AnosExperiencia: float = Field(..., example=10.0)
    AnosEmpresa: float = Field(..., example=5.0)
    AnosCargoAtual: float = Field(..., example=3.0)
    AnosUltimaPromocao: float = Field(..., example=2.0)
    TempoSemPromocao: float = Field(..., example=2.0)
    ViagemTrabalho: Literal["Não viaja", "Às vezes", "Frequentemente"]
    HorasExtras: float = Field(..., example=12.0)
    AreaFormacao: Literal["TI", "Engenharia", "Administração", "RH", "Outros"]
    SatisfacaoTrabalho: int = Field(..., ge=1, le=5, example=4)
    SatisfacaoAmbiente: int = Field(..., ge=1, le=5, example=3)
    SatisfacaoRelacionamento: int = Field(..., ge=1, le=5, example=4)
    EquilibrioVida: int = Field(..., ge=1, le=5, example=3)
    EmpresasAnteriores: int = Field(..., example=2)
    AnosGestorAtual: float = Field(..., example=2.0)
    DistanciaCasa: float = Field(..., example=8.0)

class ListaFuncionarios(BaseModel):
    funcionarios: List[Funcionario]

# ========== Rotas ==========
@app.get("/")
def home():
    return {"mensagem": "API de previsão de desligamento ativa com sucesso."}

@app.post("/prever")
def prever_desligamento(entrada: ListaFuncionarios):
    try:
        # Converter entrada em DataFrame
        dados = pd.DataFrame([f.dict() for f in entrada.funcionarios])
        
        # Preparar dados com feature engineering e codificação
        dados = preparar_dados(dados)
        
        # Aplicar pré-processamento
        dados_transf = preprocessador.transform(dados)
        
        # Fazer previsão
        probabilidades = modelo.predict_proba(dados_transf)[:, 1]
        
        # Resposta
        return {
            "resultados": [
                {
                    "indice": i,
                    "probabilidade_desligamento": round(float(prob), 4)
                }
                for i, prob in enumerate(probabilidades)
            ]
        }

    except Exception as e:
        return {"erro": str(e)}
