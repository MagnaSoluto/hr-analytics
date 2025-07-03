import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
import optuna
from optuna.samplers import TPESampler

def gerar_base_rh_analitico(qtd_amostras=1_000_000, semente=42):
    """
    Gera um dataset sintético baseado em empresas brasileiras para análise de rotatividade de funcionários (desligamento).

    Parâmetros:
    - qtd_amostras (int): número de funcionários simulados
    - semente (int): semente para reprodutibilidade

    Retorna:
    - pd.DataFrame: dataframe com as colunas simuladas
    """
    np.random.seed(semente)
    dados = {}

    # Faixa etária mais realista no Brasil (com foco entre 25 e 50 anos)
    dados['Idade'] = np.clip(np.random.normal(36, 9, qtd_amostras).astype(int), 18, 65)

    # Gênero
    dados['Genero'] = np.random.choice(['Feminino', 'Masculino'], qtd_amostras, p=[0.45, 0.55])

    # Escolaridade (1-Fundamental, 2-Médio, 3-Técnico, 4-Superior, 5-Pós)
    escolaridade_probs = [0.08, 0.25, 0.20, 0.35, 0.12]
    dados['Escolaridade'] = np.random.choice([1, 2, 3, 4, 5], qtd_amostras, p=escolaridade_probs)

    # Área de formação
    areas_formacao = ['Exatas', 'Humanas', 'Saúde', 'Marketing', 'Tecnologia', 'Recursos Humanos']
    dados['AreaFormacao'] = np.random.choice(areas_formacao, qtd_amostras)

    # Setor
    setor_probs = [0.50, 0.35, 0.15]  # Vendas mais comum, RH menos comum
    dados['Setor'] = np.random.choice(['Vendas', 'Pesquisa e Desenvolvimento', 'Recursos Humanos'],
                                      qtd_amostras, p=setor_probs)

    # Cargo baseado no setor
    cargos = {
        'Vendas': ['Executivo de Vendas', 'Representante Comercial', 'Gerente de Contas'],
        'Pesquisa e Desenvolvimento': ['Pesquisador', 'Técnico de Laboratório', 'Diretor de P&D', 'Gerente de Projeto'],
        'Recursos Humanos': ['Analista de RH', 'Coordenador de RH']
    }

    dados['Cargo'] = np.empty(qtd_amostras, dtype=object)
    for setor in cargos:
        mask = np.array(dados['Setor']) == setor
        n_setor = mask.sum()
        if n_setor > 0:
            dados['Cargo'][mask] = np.random.choice(cargos[setor], n_setor)

    # Nível do cargo
    base = np.ones(qtd_amostras)
    idade_bonus = (np.array(dados['Idade']) - 18) / 47 * 2
    esc_bonus = (np.array(dados['Escolaridade']) - 1) / 4 * 2
    dados['NivelCargo'] = np.clip(np.round(base + idade_bonus + esc_bonus).astype(int), 1, 5)

    # Tempo de experiência
    dados['AnosExperiencia'] = np.maximum(0, dados['Idade'] - 18 - np.random.randint(0, 5, qtd_amostras))

    # Tempo na empresa
    dados['AnosEmpresa'] = np.minimum(np.random.randint(0, 16, qtd_amostras), dados['AnosExperiencia'])

    # Tempo no cargo atual
    dados['AnosCargoAtual'] = np.minimum(np.random.randint(0, 11, qtd_amostras), dados['AnosEmpresa'])

    # Tempo desde a última promoção
    dados['AnosUltimaPromocao'] = np.minimum(np.random.randint(0, 7, qtd_amostras), dados['AnosEmpresa'])

    # Tempo com o gestor atual
    dados['AnosGestorAtual'] = np.minimum(np.random.randint(0, 7, qtd_amostras), dados['AnosCargoAtual'])

    # Empresas anteriores
    max_empresas = np.minimum(dados['AnosExperiencia'] // 2, 8)
    dados['EmpresasAnteriores'] = [np.random.randint(0, max(1, mc) + 1) for mc in max_empresas]

    # Salário mensal (ajustado à realidade brasileira)
    salario_base = 1800
    fator_nivel = dados['NivelCargo'] * 2200
    fator_esc = dados['Escolaridade'] * 400
    fator_exp = dados['AnosExperiencia'] * 120
    ruido = np.random.normal(0, 900, qtd_amostras)

    dados['SalarioMensal'] = np.clip((salario_base + fator_nivel + fator_esc + fator_exp + ruido).astype(int), 1400, 20000)

    # Distância de casa até o trabalho
    dados['DistanciaCasa'] = np.clip(np.random.exponential(6, qtd_amostras).astype(int) + 1, 1, 35)

    # Viagem a trabalho
    viagens_probs = [0.68, 0.22, 0.10]
    dados['ViagemTrabalho'] = np.random.choice(['Raramente', 'Frequente', 'Nunca'], qtd_amostras, p=viagens_probs)

    # Satisfação e equilíbrio
    dados['SatisfacaoAmbiente'] = np.random.choice([1, 2, 3, 4], qtd_amostras, p=[0.08, 0.20, 0.45, 0.27])
    dados['SatisfacaoTrabalho'] = np.random.choice([1, 2, 3, 4], qtd_amostras, p=[0.09, 0.21, 0.43, 0.27])
    dados['SatisfacaoRelacionamento'] = np.random.choice([1, 2, 3, 4], qtd_amostras, p=[0.10, 0.20, 0.40, 0.30])
    dados['EquilibrioVida'] = np.random.choice([1, 2, 3, 4], qtd_amostras, p=[0.10, 0.25, 0.45, 0.20])

    # Horas extras
    prob_horas_extras = 0.25 + (65 - dados['Idade']) / 47 * 0.1 + (5 - dados['NivelCargo']) / 4 * 0.1
    prob_horas_extras = np.clip(prob_horas_extras, 0.10, 0.60)
    dados['HorasExtras'] = [np.random.choice(['Sim', 'Não'], p=[p, 1 - p]) for p in prob_horas_extras]

    # Estado civil
    dados['EstadoCivil'] = np.random.choice(['Solteiro(a)', 'Casado(a)', 'Divorciado(a)'], qtd_amostras, p=[0.33, 0.53, 0.14])

    # Score de desligamento
    score_desligamento = np.zeros(qtd_amostras)
    score_desligamento += (np.array(dados['SatisfacaoTrabalho']) == 1) * 0.15
    score_desligamento += (np.array(dados['SatisfacaoAmbiente']) == 1) * 0.12
    score_desligamento += (np.array(dados['EquilibrioVida']) == 1) * 0.10
    score_desligamento += (np.array(dados['HorasExtras']) == 'Sim') * 0.08
    score_desligamento += (np.array(dados['AnosUltimaPromocao']) > 5) * 0.04
    score_desligamento += (np.array(dados['DistanciaCasa']) > 20) * 0.05
    score_desligamento += (np.array(dados['EstadoCivil']) == 'Solteiro(a)') * 0.03
    score_desligamento += (np.array(dados['EmpresasAnteriores']) > 4) * 0.04
    score_desligamento -= (np.array(dados['NivelCargo']) >= 4) * 0.09
    score_desligamento -= (np.array(dados['AnosEmpresa']) > 10) * 0.07

    prob_desligamento = np.clip(0.15 + score_desligamento, 0.05, 0.45)
    aleatorio = np.random.rand(qtd_amostras)
    dados['Desligamento'] = np.where(aleatorio < prob_desligamento, 'Sim', 'Não')

    return pd.DataFrame(dados)


def preparar_dados_rh(qtd_amostras=100_000, semente=42):
    """
    Gera a base sintética com feature engineering, codificação, escalonamento e split treino/teste.

    Parâmetros:
    - qtd_amostras (int): número de registros a gerar.
    - semente (int): semente para reprodutibilidade.

    Retorna:
    - df_encoded (pd.DataFrame): base com todas as transformações.
    - X_train, X_test, y_train, y_test: conjuntos prontos para modelagem.
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    # 1. Geração da base
    df = gerar_base_rh_analitico(qtd_amostras=qtd_amostras, semente=semente)

    # 2. Novas features
    df['SatisfacaoMedia'] = df[['SatisfacaoTrabalho', 'SatisfacaoAmbiente', 'SatisfacaoRelacionamento']].mean(axis=1)
    df['FaixaEtaria'] = pd.cut(df['Idade'], bins=[17, 29, 45, 66], labels=['<30', '30-45', '>45'])
    df['DistanteDoTrabalho'] = (df['DistanciaCasa'] > 20).astype(int)
    df['ViajaMuito'] = (df['ViagemTrabalho'] == 'Frequente').astype(int)
    df['FormacaoSuperiorOuMais'] = (df['Escolaridade'] >= 4).astype(int)
    df['RiscoHorasExtras'] = ((df['HorasExtras'] == 'Sim') & (df['EquilibrioVida'] <= 2)).astype(int)
    df['TempoSemPromocao'] = df['AnosUltimaPromocao']
    df['ExperienciaPorEmpresa'] = df['AnosExperiencia'] / (df['EmpresasAnteriores'] + 1)
    df['SalarioAjustado'] = df['SalarioMensal'] / df['NivelCargo'].replace(0, 1)
    df['EstabilidadeNaEmpresa'] = df['AnosEmpresa'] / df['AnosExperiencia'].replace(0, 1)

    # 3. Codificação
    df_encoded = df.copy()
    label_cols = ['Genero', 'AreaFormacao', 'Setor', 'Cargo', 'ViagemTrabalho',
                  'HorasExtras', 'EstadoCivil', 'Desligamento', 'FaixaEtaria']
    label_encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # 4. Escalonamento
    num_cols = ['Idade', 'AnosExperiencia', 'AnosEmpresa', 'AnosCargoAtual', 'AnosUltimaPromocao',
                'AnosGestorAtual', 'EmpresasAnteriores', 'SalarioMensal', 'DistanciaCasa',
                'SatisfacaoMedia', 'ExperienciaPorEmpresa', 'SalarioAjustado', 'EstabilidadeNaEmpresa']
    scaler = StandardScaler()
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

    # 5. Split
    X = df_encoded.drop('Desligamento', axis=1)
    y = df_encoded['Desligamento']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=semente
    )

    return df_encoded, X_train, X_test, y_train, y_test

def avaliar_modelos_thresholds(qtd_amostras=100_000, thresholds=np.arange(0.1, 0.91, 0.1), semente=42):
    """
    Realiza avaliação preditiva de modelos com ajuste de desbalanceamento e thresholds dinâmicos.

    Esta função aplica três algoritmos de classificação (Regressão Logística, Random Forest e LightGBM)
    sobre a base de dados sintética gerada pela função `preparar_dados_rh`, treinando e testando os modelos
    com ajuste para classes desbalanceadas.

    Para cada modelo, é avaliado o desempenho com diferentes limiares (thresholds) de decisão,
    imprimindo métricas como Precision, Recall, F1-Score e ROC AUC.

    Parâmetros:
    ----------
    qtd_amostras : int, padrão=100_000
        Número de amostras sintéticas a serem geradas para o experimento.

    thresholds : np.array, padrão=np.arange(0.1, 0.91, 0.1)
        Lista de limiares para avaliar a sensibilidade do modelo na classe positiva.

    semente : int, padrão=42
        Semente para garantir reprodutibilidade dos resultados.

    Retorno:
    -------
    Apenas imprime os resultados no console (não há retorno direto).
    """
    # Preparar base com novas features e pré-processamento
    df_encoded, X_train, X_test, y_train, y_test = preparar_dados_rh(qtd_amostras=qtd_amostras, semente=semente)

    # Modelos com ajustes de balanceamento
    models = {
        'Regressao Logistica': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=semente),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=semente),
        'LightGBM': lgb.LGBMClassifier(is_unbalance=True, random_state=semente)
    }

    # Avaliação
    for nome, modelo in models.items():
        print(f"\n📌 Modelo: {nome}")
        modelo.fit(X_train, y_train)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)

        print("Threshold | Precision | Recall | F1-Score | ROC AUC")
        print("-" * 55)
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            print(f"{t:9.2f} | {precision:9.2f} | {recall:6.2f} | {f1:8.2f} | {roc_auc:.4f}")

def tunar_modelo_optuna(modelo_nome, X_train, y_train, n_trials=50, semente=42, timeout=None):
    """
    Tuna um modelo de machine learning utilizando a biblioteca Optuna para otimização de hiperparâmetros.
    Parâmetros:
    -----------
    modelo_nome : str
        O nome do modelo a ser otimizado. Aceita 'lightgbm', 'random_forest' ou 'logistic_regression'.
    X_train : pd.DataFrame
        Conjunto de dados de treinamento (features).
    y_train : pd.Series
        Conjunto de dados de treinamento (target).
    n_trials : int, opcional
        Número de tentativas para a otimização. O padrão é 50.
    semente : int, opcional
        Semente para reprodutibilidade. O padrão é 42.
    timeout : int, opcional
        Tempo máximo em segundos para a otimização. O padrão é None, o que significa sem limite.
    Retorna:
    --------
    best_model : sklearn.base.BaseEstimator
        O modelo treinado com os melhores hiperparâmetros encontrados.
    study : optuna.study.Study
        O objeto de estudo do Optuna contendo informações sobre a otimização.
    Levanta:
    --------
    ValueError
        Se o modelo especificado não for suportado.
    Exemplo:
    --------
    >>> best_model, study = tunar_modelo_optuna('lightgbm', X_train, y_train)
    """

    def objective(trial):
        if modelo_nome == 'lightgbm':
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 10, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
                "n_estimators": 1000,
                "objective": "binary",
                "is_unbalance": True,
                "random_state": semente,
                "verbosity": -1
            }

        elif modelo_nome == 'random_forest':
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ['sqrt', 'log2', None]),
                "class_weight": "balanced",
                "n_jobs": -1,
                "random_state": semente
            }

        elif modelo_nome == 'logistic_regression':
            params = {
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "solver": "liblinear",
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": semente
            }

        else:
            raise ValueError("Modelo não suportado. Use: 'lightgbm', 'random_forest' ou 'logistic_regression'.")

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=semente)
        scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            if modelo_nome == 'lightgbm':
                model = lgb.LGBMClassifier(**params)
                model.fit(X_tr, y_tr)
                y_pred = model.predict_proba(X_val)[:, 1]

            else:
                if modelo_nome == 'random_forest':
                    model = RandomForestClassifier(**params)
                elif modelo_nome == 'logistic_regression':
                    model = LogisticRegression(**params)
                model.fit(X_tr, y_tr)
                y_pred = model.predict_proba(X_val)[:, 1]

            score = roc_auc_score(y_val, y_pred)
            scores.append(score)

        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=semente), study_name=f"Tuning_{modelo_nome}")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    print("\n✅ Melhor AUC: {:.4f}".format(study.best_value))
    print("🏆 Melhor conjunto de hiperparâmetros:")
    for k, v in study.best_params.items():
        print(f"   - {k}: {v}")

    if modelo_nome == 'lightgbm':
        best_model = lgb.LGBMClassifier(**study.best_params)
    elif modelo_nome == 'random_forest':
        best_model = RandomForestClassifier(**study.best_params)
    elif modelo_nome == 'logistic_regression':
        best_model = LogisticRegression(**study.best_params)

    best_model.fit(X_train, y_train)

    return best_model, study

def criar_variaveis_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variáveis derivadas a partir do dataframe de funcionários.
    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo as informações dos funcionários.
    Retorna:
    - pd.DataFrame: DataFrame com as novas variáveis derivadas.
    Esta função adiciona as seguintes variáveis ao DataFrame:
    - SatisfacaoMedia: Média das satisfações de trabalho, ambiente e relacionamento.
    - FaixaEtaria: Categorização da idade em faixas etárias.
    - DistanteDoTrabalho: Indicador se a distância de casa é maior que 20 km.
    - ViajaMuito: Indicador se o funcionário viaja frequentemente a trabalho.
    - FormacaoSuperiorOuMais: Indicador se o funcionário possui formação superior ou mais.
    - RiscoHorasExtras: Indicador se o funcionário corre risco de horas extras excessivas.
    - TempoSemPromocao: Anos desde a última promoção.
    """
    df['ExperienciaPorEmpresa'] = df['AnosExperiencia'] / df['EmpresasAnteriores'].replace(0, 1)
    df['FormacaoSuperiorOuMais'] = df['AreaFormacao'].apply(lambda x: 1 if x in ['TI', 'Engenharia', 'RH'] else 0)
    df['DistanteDoTrabalho'] = df['DistanciaCasa'].apply(lambda x: 1 if x > 15 else 0)
    df['SalarioAjustado'] = df['SalarioMensal'] / df['NivelCargo'].map({
        'Júnior': 1, 'Pleno': 2, 'Sênior': 3, 'Gerente': 4
    }).replace(0, 1)

    df['SatisfacaoMedia'] = df[[
        'SatisfacaoTrabalho', 'SatisfacaoAmbiente',
        'SatisfacaoRelacionamento', 'EquilibrioVida'
    ]].mean(axis=1)

    df['EstabilidadeNaEmpresa'] = df['AnosEmpresa'] - df['EmpresasAnteriores']
    df['ViajaMuito'] = df['ViagemTrabalho'].apply(lambda x: 1 if x == 'Frequentemente' else 0)
    df['RiscoHorasExtras'] = df['HorasExtras'].apply(lambda x: 'Sim' if x > 10 else 'Não')

    # FaixaEtaria como categórica
    df['FaixaEtaria'] = pd.cut(
        df['Idade'],
        bins=[0, 25, 35, 45, 55, 100],
        labels=['Até 25', '26-35', '36-45', '46-55', 'Acima de 55']
    ).astype(str)

    return df
