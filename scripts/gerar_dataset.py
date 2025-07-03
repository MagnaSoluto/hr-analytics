import os
import pandas as pd
import sys

# Adicionar raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.func import gerar_base_rh_analitico

def main():
    print("Gerando dataset sintético de RH...")

    # Gerar dataset
    df = gerar_base_rh_analitico(qtd_amostras=100_000, semente=42)

    # Garantir que a pasta data/ existe
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_path, exist_ok=True)

    # Salvar como CSV
    csv_path = os.path.join(data_path, 'dataset.csv')
    df.to_csv(csv_path, index=False)

    print(f"✅ Dataset gerado com sucesso em: {csv_path}")

if __name__ == '__main__':
    main()
