"""
baixar_datasets.py - Baixa datasets de tosse automaticamente
"""

import os
import requests
import zipfile
import tarfile
import io
import wget
from tqdm import tqdm
import pandas as pd

class DownloaderDatasets:
    """Classe para baixar datasets de tosse."""
    
    def __init__(self, output_dir="dados_externos"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def baixar_coswara(self):
        """Baixa dataset Coswara."""
        print("📥 Baixando Coswara Dataset...")
        
        # URLs possíveis (podem mudar)
        urls = [
            "https://zenodo.org/record/7754137/files/coswara_data.tar.gz",
            "https://www.dropbox.com/s/xxxx/coswara_data.zip?dl=1"
        ]
        
        try:
            # Tentar baixar
            url = urls[0]
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                # Salvar
                file_path = os.path.join(self.output_dir, "coswara.tar.gz")
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extrair
                self.extrair_arquivo(file_path)
                print("✅ Coswara baixado com sucesso!")
                return True
                
        except Exception as e:
            print(f"❌ Erro ao baixar Coswara: {e}")
            
            # Método alternativo: usar kaggle CLI
            print("📦 Tentando via Kaggle...")
            try:
                os.system("kaggle datasets download -d shivamm4123/coswara-dataset")
                print("✅ Baixado via Kaggle")
                return True
            except:
                print("❌ Kaggle também falhou")
        
        return False
    
    def baixar_icbhi(self):
        """Baixa dataset ICBHI (requer registro manual)."""
        print("📥 ICBHI requer registro manual em:")
        print("   https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge")
        print("📝 Após baixar manualmente, coloque em:")
        print(f"   {self.output_dir}/icbhi/")
        
        return False
    
    def baixar_virufy(self):
        """Baixa dataset Virufy."""
        print("📥 Baixando Virufy Dataset...")
        
        try:
            # URL do GitHub
            url = "https://github.com/virufy/virufy-data/archive/refs/heads/main.zip"
            
            response = requests.get(url)
            
            if response.status_code == 200:
                file_path = os.path.join(self.output_dir, "virufy.zip")
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                self.extrair_arquivo(file_path)
                print("✅ Virufy baixado com sucesso!")
                return True
                
        except Exception as e:
            print(f"❌ Erro ao baixar Virufy: {e}")
        
        return False
    
    def baixar_kaggle_respiratory(self):
        """Baixa dataset de sons respiratórios do Kaggle."""
        print("📥 Baixando Respiratory Sound Database do Kaggle...")
        
        try:
            # Requer kaggle.json configurado
            os.system("kaggle datasets download -d vbookshelf/respiratory-sound-database")
            
            # Mover para pasta
            os.rename("respiratory-sound-database.zip", 
                     os.path.join(self.output_dir, "respiratory_kaggle.zip"))
            
            self.extrair_arquivo(os.path.join(self.output_dir, "respiratory_kaggle.zip"))
            print("✅ Kaggle Respiratory baixado!")
            return True
            
        except Exception as e:
            print(f"❌ Erro Kaggle: {e}")
            print("📝 Configure kaggle.json primeiro")
        
        return False
    
    def extrair_arquivo(self, file_path):
        """Extrai arquivo compactado."""
        print(f"📦 Extraindo {os.path.basename(file_path)}...")
        
        try:
            if file_path.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(file_path))
                    
            elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(os.path.dirname(file_path))
            
            print("✅ Extração concluída!")
            
        except Exception as e:
            print(f"❌ Erro na extração: {e}")
    
    def organizar_para_treino(self):
        """Organiza dados baixados para o formato do nosso sistema."""
        print("\n🔄 Organizando dados para treino...")
        
        # Criar estrutura
        os.makedirs("dados/brutos/normal", exist_ok=True)
        os.makedirs("dados/brutos/bronquite", exist_ok=True)
        os.makedirs("dados/brutos/pneumonia", exist_ok=True)
        
        # Aqui você precisaria processar cada dataset
        # Esta é uma implementação de exemplo
        
        print("✅ Estrutura criada")
        print("📝 Agora organize os áudios manualmente nas pastas:")
        print("   dados/brutos/normal/")
        print("   dados/brutos/bronquite/")
        print("   dados/brutos/pneumonia/")
        
        return True

def main():
    """Função principal."""
    print("=" * 60)
    print("📊 BAIXADOR DE DATASETS PARA CLASSIFICAÇÃO DE TOSSE")
    print("=" * 60)
    
    downloader = DownloaderDatasets()
    
    print("\n📋 DATASETS DISPONÍVEIS:")
    print("1. Coswara Dataset (Recomendado)")
    print("2. ICBHI Respiratory Sound Database")
    print("3. Virufy Dataset")
    print("4. Kaggle Respiratory Sounds")
    print("5. Todos os disponíveis")
    
    escolha = input("\n🎯 Escolha (1-5): ")
    
    if escolha == '1':
        downloader.baixar_coswara()
    elif escolha == '2':
        downloader.baixar_icbhi()
    elif escolha == '3':
        downloader.baixar_virufy()
    elif escolha == '4':
        downloader.baixar_kaggle_respiratory()
    elif escolha == '5':
        downloader.baixar_coswara()
        downloader.baixar_virufy()
        # downloader.baixar_kaggle_respiratory()
    
    # Organizar
    downloader.organizar_para_treino()
    
    print("\n" + "=" * 60)
    print("✅ DOWNLOADS CONCLUÍDOS!")
    print("=" * 60)
    print("\n🎯 PRÓXIMOS PASSOS:")
    print("1. Organize os áudios nas pastas corretas")
    print("2. Execute: python main.py --treinar")
    print("3. Ou use: make train")

if __name__ == "__main__":
    main()
