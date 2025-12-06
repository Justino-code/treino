"""
verificar_dados.py - Verifica se os dados estão corretos
"""

import os
import glob

def verificar_estrutura():
    """Verifica estrutura de pastas."""
    print("🔍 VERIFICANDO ESTRUTURA...")
    
    pastas_necessarias = [
        'dados/brutos/normal',
        'dados/brutos/bronquite',
        'dados/brutos/pneumonia'
    ]
    
    tudo_ok = True
    
    for pasta in pastas_necessarias:
        if os.path.exists(pasta):
            print(f"✅ {pasta}/")
        else:
            print(f"❌ {pasta}/ (NÃO ENCONTRADA)")
            tudo_ok = False
    
    return tudo_ok

def contar_audios():
    """Conta áudios em cada pasta."""
    print("\n🎵 CONTANDO ÁUDIOS...")
    
    classes = ['normal', 'bronquite', 'pneumonia']
    total = 0
    
    for classe in classes:
        pasta = f'dados/brutos/{classe}'
        
        if not os.path.exists(pasta):
            print(f"❌ {classe}: Pasta não existe")
            continue
        
        # Contar arquivos de áudio
        extensoes = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']
        contagem = 0
        
        for ext in extensoes:
            contagem += len(glob.glob(os.path.join(pasta, ext)))
        
        print(f"📊 {classe}: {contagem} áudios")
        total += contagem
    
    print(f"\n🎯 TOTAL: {total} áudios")
    
    if total == 0:
        print("❌ ERRO: Nenhum áudio encontrado!")
        return False
    
    # Recomendações
    if total < 30:
        print("⚠️  AVISO: Poucos dados para treinamento!")
        print("   Recomendado: pelo menos 50-100 áudios por classe")
    
    return True

def verificar_formatos():
    """Verifica formatos dos arquivos."""
    print("\n📁 VERIFICANDO FORMATOS...")
    
    classes = ['normal', 'bronquite', 'pneumonia']
    formatos = {}
    
    for classe in classes:
        pasta = f'dados/brutos/{classe}'
        
        if not os.path.exists(pasta):
            continue
        
        for arquivo in os.listdir(pasta):
            ext = os.path.splitext(arquivo)[1].lower()
            formatos[ext] = formatos.get(ext, 0) + 1
    
    if formatos:
        print("Formatos encontrados:")
        for ext, quantidade in formatos.items():
            print(f"  {ext}: {quantidade} arquivos")
        
        # Verificar formatos suportados
        suportados = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        problemas = []
        
        for ext in formatos.keys():
            if ext not in suportados:
                problemas.append(ext)
        
        if problemas:
            print(f"\n⚠️  AVISO: Formatos não otimizados encontrados: {problemas}")
            print("   Recomendado: converter para .wav ou .mp3")
    
    return True

def mostrar_comandos():
    """Mostra próximos passos."""
    print("\n" + "=" * 60)
    print("🚀 PRÓXIMOS PASSOS:")
    print("=" * 60)
    
    if not os.path.exists('dados/brutos/normal'):
        print("1. 📁 Crie a estrutura de pastas:")
        print("   python preparar_dados.py")
    
    # Contar áudios
    total = 0
    for classe in ['normal', 'bronquite', 'pneumonia']:
        pasta = f'dados/brutos/{classe}'
        if os.path.exists(pasta):
            audios = glob.glob(os.path.join(pasta, '*.*'))
            total += len([a for a in audios if os.path.splitext(a)[1].lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']])
    
    if total == 0:
        print("\n2. 🎵 Adicione áudios nas pastas ou use:")
        print("   python preparar_dados.py")
    elif total < 30:
        print(f"\n2. ⚠️  Você tem apenas {total} áudios")
        print("   Adicione mais dados para melhorar o modelo")
        print("   Continuar mesmo assim? (python main_simples.py)")
    else:
        print(f"\n2. ✅ Você tem {total} áudios - Ótimo!")
        print("   Execute o treinamento:")
        print("   python main_simples.py")
    
    print("\n3. 📱 Para usar no celular:")
    print("   Após treinar, veja resultados/COMO_USAR_NO_APP.txt")

def main():
    """Função principal."""
    print("=" * 60)
    print("🔍 VERIFICADOR DE DADOS DE TOSSE")
    print("=" * 60)
    
    # Verificar estrutura
    estrutura_ok = verificar_estrutura()
    
    if not estrutura_ok:
        print("\n❌ Estrutura incompleta!")
        print("Execute: python preparar_dados.py")
        return
    
    # Verificar áudios
    dados_ok = contar_audios()
    
    if not dados_ok:
        print("\n❌ Sem áudios para treinar!")
        return
    
    # Verificar formatos
    verificar_formatos()
    
    # Mostrar próximos passos
    mostrar_comandos()

if __name__ == "__main__":
    main()
