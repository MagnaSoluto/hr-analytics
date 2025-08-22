import os
import pandas as pd
import sys

# Adicionar raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.func import gerar_base_rh_analitico

def main():
    print("Gerando dataset sintético de RH...")

    # Gerar dataset garantindo desbalanceamento ~16%
    df = gerar_base_rh_analitico(qtd_amostras=100_000, semente=42)

    # Verificar desbalanceamento real
    perc_attrition = df['Desligamento'].value_counts(normalize=True).get('Sim', 0) * 100
    print(f"📌 Proporção de attrition no dataset gerado: {perc_attrition:.2f}%")
    
    if abs(perc_attrition - 16) > 5:
        raise ValueError("⚠️ O dataset não está com o desbalanceamento correto (~16% attrition).")

    # Conferir número de variáveis
    num_variaveis = df.shape[1]
    print(f"📌 Número de variáveis geradas: {num_variaveis}")
    
    if num_variaveis < 25:
        raise ValueError(f"⚠️ Dataset gerado com menos variáveis do que o necessário (esperado: 25, obtido: {num_variaveis}).")

    # Garantir que a pasta data/ existe
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_path, exist_ok=True)

    # Salvar como CSV
    csv_path = os.path.join(data_path, 'dataset.csv')
    df.to_csv(csv_path, index=False)

    print(f"✅ Dataset gerado com sucesso em: {csv_path}")

if __name__ == '__main__':
    main()
