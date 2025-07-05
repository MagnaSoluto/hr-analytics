import os
import sys

# Adicionar raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.func import gerar_base_rh_analitico


def main():
    print("Gerando dados para treinamento...")
    df = gerar_base_rh_analitico(qtd_amostras=100_000, semente=42)
    print(df.head())


if __name__ == '__main__':
    main()
