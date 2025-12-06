# criar_dataset_teste.py
import numpy as np
import soundfile as sf
import os

def criar_dataset_teste():
    """Cria dataset de teste simulado."""
    
    classes = ['normal', 'bronquite', 'pneumonia']
    sr = 16000  # Taxa de amostragem
    duracao = 3  # segundos
    
    for classe in classes:
        os.makedirs(f'dados/brutos/{classe}', exist_ok=True)
        
        for i in range(10):  # 10 áudios por classe
            # Criar áudio sintético
            t = np.linspace(0, duracao, int(sr * duracao))
            
            if classe == 'normal':
                # Tosse "normal" - ruído suave
                audio = np.random.randn(len(t)) * 0.1
                # Adicionar algum padrão
                for freq in [100, 200, 300]:
                    audio += 0.05 * np.sin(2 * np.pi * freq * t)
                    
            elif classe == 'bronquite':
                # Tosse seca - picos agudos
                audio = np.random.randn(len(t)) * 0.15
                # Picos periódicos
                for j in range(5):
                    inicio = j * 0.5
                    idx = (t >= inicio) & (t < inicio + 0.1)
                    audio[idx] += 0.3 * np.sin(2 * np.pi * 500 * t[idx])
                    
            else:  # pneumonia
                # Tosse produtiva - padrão complexo
                audio = np.random.randn(len(t)) * 0.12
                # Múltiplas frequências
                for freq in [80, 120, 180, 250]:
                    amplitude = 0.8 / freq
                    audio += amplitude * np.sin(2 * np.pi * freq * t)
            
            # Normalizar
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            # Salvar
            sf.write(f'dados/brutos/{classe}/tosse_{classe}_{i+1:02d}.wav', audio, sr)
    
    print("✅ Dataset de teste criado!")
    print("📁 Estrutura:")
    print("   dados/brutos/normal/ - 10 áudios")
    print("   dados/brutos/bronquite/ - 10 áudios")
    print("   dados/brutos/pneumonia/ - 10 áudios")
    print("\n🚀 Agora execute: python main.py --treinar")

if __name__ == "__main__":
    criar_dataset_teste()
