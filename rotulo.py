import os
import csv

# Caminho para a pasta com os dados brutos
caminho_dados = 'dados/brutos'

# Caminho para salvar o arquivo de rótulos
arquivo_rotulos = 'dados/rotulos.csv'

# Abrindo o arquivo CSV para escrita
with open(arquivo_rotulos, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # Escrevendo o cabeçalho
    writer.writerow(['arquivo', 'rotulo'])

    # Percorrendo as subpastas
    for rotulo in os.listdir(caminho_dados):
        pasta_rotulo = os.path.join(caminho_dados, rotulo)
        if os.path.isdir(pasta_rotulo):
            # Para cada arquivo dentro da pasta
            for nome_arquivo in os.listdir(pasta_rotulo):
                caminho_arquivo = os.path.join(rotulo, nome_arquivo)  # caminho relativo
                writer.writerow([caminho_arquivo, rotulo])

print(f"Arquivo {arquivo_rotulos} criado com sucesso!")
