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

# Função para criar variáveis de engenharia
def adicionar_variaveis_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['SatisfacaoMedia'] = df[['SatisfacaoTrabalho', 'SatisfacaoAmbiente', 'SatisfacaoRelacionamento']].mean(axis=1)
    df['FormacaoSuperiorOuMais'] = df['AreaFormacao'].apply(lambda x: 1 if x in ['TI', 'Engenharia', 'Administração'] else 0)
    df['DistanteDoTrabalho'] = (df['DistanciaCasa'] > 10).astype(int)
    df['RiscoHorasExtras'] = (df['HorasExtras'] > 10).astype(int)
    df['ViajaMuito'] = df['ViagemTrabalho'].apply(lambda x: 1 if x == 'Frequentemente' else 0)
    df['SalarioAjustado'] = df['SalarioMensal'] / (df['AnosExperiencia'] + 1)
    df['ExperienciaPorEmpresa'] = df['AnosExperiencia'] / (df['EmpresasAnteriores'] + 1)
    df['EstabilidadeNaEmpresa'] = df['AnosEmpresa'] / (df['EmpresasAnteriores'] + 1)
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
        
        # Criar colunas adicionais
        dados = adicionar_variaveis_engineering(dados)
        
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
