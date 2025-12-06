"""
preparar_dados.py - Prepara dados automaticamente

COMO USAR:
1. Coloque TODOS seus áudios em qualquer lugar
2. Execute: python preparar_dados.py
3. O sistema organiza automaticamente nas pastas corretas
"""

import os
import shutil
import glob
import random

def criar_estrutura():
    """Cria estrutura de pastas."""
    pastas = [
        'dados/brutos/normal',
        'dados/brutos/bronquite', 
        'dados/brutos/pneumonia',
        'dados/nao_classificados'
    ]
    
    for pasta in pastas:
        os.makedirs(pasta, exist_ok=True)
    
    print("✅ Pastas criadas:")
    for pasta in pastas:
        print(f"   📁 {pasta}")

def encontrar_audios():
    """Encontra todos os arquivos de áudio."""
    extensoes = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg', '*.WAV', '*.MP3']
    audios = []
    
    for ext in extensoes:
        audios.extend(glob.glob(f"**/{ext}", recursive=True))
    
    # Remover áudios que já estão em dados/brutos/
    audios = [a for a in audios if 'dados/brutos' not in a]
    
    return audios

def organizar_manual(audios):
    """Organiza áudios manualmente (para classificação)."""
    print(f"\n🔍 Encontrados {len(audios)} áudios não classificados")
    print("Vamos organizá-los!")
    
    for i, audio_path in enumerate(audios, 1):
        print(f"\n🎵 Áudio {i}/{len(audios)}: {os.path.basename(audio_path)}")
        print("   Qual a classe deste áudio?")
        print("   1. Normal")
        print("   2. Bronquite")
        print("   3. Pneumonia")
        print("   4. Ignorar")
        
        while True:
            try:
                escolha = input("   Digite o número (1-4): ")
                if escolha == '1':
                    destino = 'dados/brutos/normal'
                    break
                elif escolha == '2':
                    destino = 'dados/brutos/bronquite'
                    break
                elif escolha == '3':
                    destino = 'dados/brutos/pneumonia'
                    break
                elif escolha == '4':
                    destino = None
                    break
                else:
                    print("   ❌ Opção inválida. Tente novamente.")
            except:
                print("   ❌ Entrada inválida.")
        
        if destino:
            # Copiar arquivo
            nome_arquivo = os.path.basename(audio_path)
            novo_caminho = os.path.join(destino, nome_arquivo)
            
            # Evitar sobreescrever
            if os.path.exists(novo_caminho):
                nome, ext = os.path.splitext(nome_arquivo)
                novo_caminho = os.path.join(destino, f"{nome}_{i}{ext}")
            
            shutil.copy2(audio_path, novo_caminho)
            print(f"   ✅ Copiado para: {destino}/")
        else:
            # Mover para não classificados
            destino = 'dados/nao_classificados'
            nome_arquivo = os.path.basename(audio_path)
            novo_caminho = os.path.join(destino, nome_arquivo)
            shutil.copy2(audio_path, novo_caminho)
            print(f"   📦 Movido para: não classificados/")

def organizar_automatico(audios, proporcao=(0.33, 0.33, 0.34)):
    """Organiza áudios automaticamente (para teste)."""
    print(f"\n🤖 Organizando {len(audios)} áudios automaticamente...")
    print("⚠️  ATENÇÃO: Esta é uma organização ALEATÓRIA, apenas para teste!")
    print("   Para dados reais, organize manualmente.")
    
    random.shuffle(audios)
    
    n_normal = int(len(audios) * proporcao[0])
    n_bronquite = int(len(audios) * proporcao[1])
    n_pneumonia = len(audios) - n_normal - n_bronquite
    
    # Distribuir
    for i, audio_path in enumerate(audios):
        if i < n_normal:
            destino = 'dados/brutos/normal'
        elif i < n_normal + n_bronquite:
            destino = 'dados/brutos/bronquite'
        else:
            destino = 'dados/brutos/pneumonia'
        
        # Copiar
        nome_arquivo = os.path.basename(audio_path)
        novo_caminho = os.path.join(destino, nome_arquivo)
        
        if os.path.exists(novo_caminho):
            nome, ext = os.path.splitext(nome_arquivo)
            novo_caminho = os.path.join(destino, f"{nome}_{i}{ext}")
        
        shutil.copy2(audio_path, novo_caminho)
    
    print(f"✅ Distribuição:")
    print(f"   Normal: {n_normal} áudios")
    print(f"   Bronquite: {n_bronquite} áudios")
    print(f"   Pneumonia: {n_pneumonia} áudios")

def main():
    """Função principal."""
    print("=" * 60)
    print("📁 ORGANIZADOR DE ÁUDIOS DE TOSSE")
    print("=" * 60)
    
    # Criar estrutura
    criar_estrutura()
    
    # Encontrar áudios
    audios = encontrar_audios()
    
    if not audios:
        print("\n❌ Nenhum áudio encontrado!")
        print("📝 Coloque arquivos .wav ou .mp3 nesta pasta ou subpastas.")
        return
    
    print("\n📋 OPÇÕES DE ORGANIZAÇÃO:")
    print("1. Organizar MANUALMENTE (recomendado para dados reais)")
    print("2. Organizar AUTOMATICAMENTE (apenas para teste)")
    print("3. Sair")
    
    escolha = input("\nEscolha (1-3): ")
    
    if escolha == '1':
        organizar_manual(audios)
    elif escolha == '2':
        organizar_automatico(audios)
    else:
        print("❌ Operação cancelada.")
        return
    
    print("\n" + "=" * 60)
    print("✅ ORGANIZAÇÃO CONCLUÍDA!")
    print("=" * 60)
    print("\n📁 SEUS ÁUDIOS ESTÃO ORGANIZADOS EM:")
    print("   dados/brutos/normal/")
    print("   dados/brutos/bronquite/")
    print("   dados/brutos/pneumonia/")
    print("\n🚀 AGORA EXECUTE: python main_simples.py")

if __name__ == "__main__":
    main()
