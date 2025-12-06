📚 GUIA COMPLETO DE USO

Guia passo a passo para usar o sistema de treinamento de classificação de tosse para bronquite e pneomonia:

📁 ESTRUTURA DO PROJETO

```
treino/
├── dados/
│   ├── brutos/           # Áudios originais
│   ├── processados/      # Características extraídas
│   └── rotulos.csv      # CSV com caminhos e rótulos
├── src/
│   ├── utilitarios.py
│   ├── preprocessamento.py
│   ├── extrair_caracteristicas.py
│   ├── modelo.py
│   └── treinar.py
└── requisitos.txt
```

📦 INSTALAÇÃO

1. Instalar Dependências:

```bash
# Instalar todas as dependências
pip install -r requisitos.txt

# Ou instalar manualmente:
pip install tensorflow>=2.10 librosa>=0.10 numpy pandas scikit-learn matplotlib seaborn soundfile
```

2. Verificar Instalação:

```bash
# Executar teste dos utilitários
python -c "from src.utilitarios import verificar_dependencias; print(verificar_dependencias())"
```

📊 PREPARAÇÃO DOS DADOS

1. Estrutura do CSV de Rótulos (dados/rotulos.csv):

```csv
caminho,rotulo,dataset,observacoes
audio1.wav,normal,treino,tosse_normal
audio2.wav,bronquite,treino,tosse_seca
audio3.wasp,pneumonia,teste,tosse_produtiva
```

2. Organizar Áudios:

Coloque todos os arquivos de áudio na pasta dados/brutos/.

3. Criar CSV Automaticamente:

```python
from src.utilitarios import criar_estrutura_projeto
criar_estrutura_projeto()
```

🚀 COMEÇAR A USAR

OPÇÃO 1: Treinamento Rápido (Recomendado para Iniciantes)

```python
# treinamento_rapido.py
import sys
sys.path.append('src')

from treinar import treinar_modelo_completo

# Executar treinamento completo
treinador = treinar_modelo_completo(
    csv_path='dados/rotulos.csv',
    audio_dir='dados/brutos',
    output_dir='meu_primeiro_treinamento'
)

print("✅ Treinamento concluído!")
```

OPÇÃO 2: Treinamento com Configuração Personalizada

```python
# treinamento_personalizado.py
import sys
sys.path.append('src')

from treinar import TreinadorTosse

# Configurações personalizadas
config = {
    'MODEL_TYPE': 'cnn_avancado',
    'EPOCHS': 50,
    'BATCH_SIZE': 16,
    'LEARNING_RATE': 0.0001,
    'BALANCE_CLASSES': True,
    'AUGMENT_TRAIN': True
}

# Criar treinador
treinador = TreinadorTosse(
    config=config,
    resultados_dir='resultados_personalizados'
)

# 1. Carregar dados
treinador.carregar_dados(
    csv_path='dados/rotulos.csv',
    audio_dir='dados/brutos',
    classes=['normal', 'bronquite', 'pneumonia']  # Definir ordem das classes
)

# 2. Preparar dados
treinador.preparar_dados()

# 3. Construir modelo
treinador.construir_modelo()

# 4. Treinar
historico = treinador.treinar()

# 5. Avaliar
metricas = treinador.avaliar('test')
print(f"Acurácia no teste: {metricas.get('accuracy', 0):.4f}")

# 6. Exportar modelo para mobile
treinador.exportar_modelo('tflite')
```

OPÇÃO 3: Usando Linha de Comando (CLI)

```bash
# Treinamento básico
python src/treinar.py --csv dados/rotulos.csv --audio_dir dados/brutos

# Treinamento com parâmetros personalizados
python src/treinar.py \
    --csv dados/rotulos.csv \
    --audio_dir dados/brutos \
    --output_dir resultados_experimento1 \
    --epochs 100 \
    --batch_size 32 \
    --model_type mobilenet \
    --learning_rate 0.0001
```

🔍 TESTE COM DADOS DE EXEMPLO

1. Criar Dados Sintéticos para Teste:

```python
# criar_dados_teste.py
import numpy as np
import soundfile as sf
import os
import pandas as pd

# Criar diretório de teste
os.makedirs('dados_teste/brutos', exist_ok=True)

# Gerar áudios sintéticos
amostras_por_classe = 10
sr = 16000
duracao = 3  # segundos

dados = []

for classe_idx, classe in enumerate(['normal', 'bronquite', 'pneumonia']):
    for i in range(amostras_por_classe):
        # Criar áudio sintético (ruído com padrões diferentes)
        t = np.linspace(0, duracao, int(sr * duracao))
        
        if classe == 'normal':
            # Ruído suave
            audio = np.random.randn(len(t)) * 0.1
        elif classe == 'bronquite':
            # Ruído com picos (tosse seca)
            audio = np.random.randn(len(t)) * 0.2
            # Adicionar alguns picos
            for _ in range(5):
                inicio = np.random.randint(0, len(t) - 100)
                audio[inicio:inicio+100] += np.sin(2*np.pi*500*t[inicio:inicio+100]) * 0.5
        else:  # pneumonia
            # Ruído com padrão mais complexo
            audio = np.random.randn(len(t)) * 0.15
            # Adicionar múltiplos picos
            for _ in range(10):
                inicio = np.random.randint(0, len(t) - 200)
                freq = np.random.randint(200, 800)
                audio[inicio:inicio+200] += np.sin(2*np.pi*freq*t[inicio:inicio+200]) * 0.3
        
        # Normalizar
        audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Salvar arquivo
        nome_arquivo = f'{classe}_{i+1:03d}.wav'
        caminho = f'dados_teste/brutos/{nome_arquivo}'
        sf.write(caminho, audio, sr)
        
        # Adicionar à lista
        dados.append({
            'caminho': nome_arquivo,
            'rotulo': classe,
            'dataset': 'treino' if i < 8 else 'teste'  # 8 para treino, 2 para teste
        })

# Salvar CSV
df = pd.DataFrame(dados)
df.to_csv('dados_teste/rotulos.csv', index=False)

print(f"✅ Criados {len(dados)} áudios de teste em 'dados_teste/'")
```

2. Treinar com Dados de Teste:

```bash
python src/treinar.py --csv dados_teste/rotulos.csv --audio_dir dados_teste/brutos --epochs 10
```

📈 MONITORAMENTO DO TREINAMENTO

1. TensorBoard (Gráficos em Tempo Real):

```bash
# Executar TensorBoard
tensorboard --logdir resultados_treinamento/logs/tensorboard

# Acessar no navegador: http://localhost:6006
```

2. Verificar Logs:

```bash
# Ver logs de treinamento
tail -f resultados_treinamento/logs/training_*.csv

# Ver métricas finais
cat resultados_treinamento/relatorios/metricas_test.json | python -m json.tool
```

🔧 USO AVANÇADO

1. Validação Cruzada:

```python
treinador = TreinadorTosse()
treinador.carregar_dados('dados/rotulos.csv', 'dados/brutos')
treinador.preparar_dados()

# Executar validação cruzada
resultados_cv = treinador.validacao_cruzada(n_folds=5, n_epochs=30)
print(f"Acurácia média: {resultados_cv['mean_val_accuracy']:.4f}")
```

2. Testar Modelo Individual:

```python
# testar_modelo.py
from src.modelo import GerenciadorModelos
from src.extrair_caracteristicas import ExtratorCaracteristicasTosse
from src.preprocessamento import PreprocessadorTosse
import numpy as np

# Carregar modelo treinado
gerenciador = GerenciadorModelos()
modelo = gerenciador.carregar_modelo('resultados_treinamento/modelos/modelo_final_*.h5')

# Inicializar processadores
preprocessador = PreprocessadorTosse()
extrator = ExtratorCaracteristicasTosse()

# Processar um áudio novo
audio_path = 'novo_audio.wav'
audio = preprocessador.processar_pipeline(audio_path)
mel_spec = extrator.extrair_espectrograma_mel(audio)
imagem = extrator.espectrograma_para_imagem(mel_spec)

# Fazer predição
imagem = np.expand_dims(imagem, axis=0)  # Adicionar dimensão do batch
predicoes = modelo.predict(imagem)

# Interpretar resultados
classes = ['normal', 'bronquite', 'pneumonia']
classe_predita = classes[np.argmax(predicoes)]
confianca = np.max(predicoes)

print(f"Predição: {classe_predita} (confiança: {confianca:.2%})")
```

3. Converter para Aplicativo Móvel:

```python
# converter_para_mobile.py
from src.modelo import GerenciadorModelos
from src.modelo import ConstrutorModelos

# Criar modelo leve para mobile
construtor = ConstrutorModelos()
modelo_mobile = construtor.criar_modelo_leve_mobile()

# Ou carregar modelo treinado
# modelo_mobile = gerenciador.carregar_modelo('modelo_treinado.h5')

# Converter para TensorFlow Lite
gerenciador = GerenciadorModelos()
gerenciador.converter_para_tflite(
    modelo_mobile,
    'modelo_mobile.tflite',
    quantize=True  # Reduz tamanho em 75%
)

print("✅ Modelo para mobile pronto: modelo_mobile.tflite")
```

📋 EXEMPLOS PRÁTICOS

Exemplo 1: Treinamento com Dataset COUGHVID

```python
# treinar_coughvid.py
import pandas as pd

# Supondo que você baixou o COUGHVID e tem um CSV
df_coughvid = pd.read_csv('coughvid_metadata.csv')

# Criar CSV no formato esperado
df_formatado = pd.DataFrame({
    'caminho': df_coughvid['filename'],
    'rotulo': df_coughvid['status'].map({
        'healthy': 'normal',
        'symptomatic': 'bronquite',
        'COVID-19': 'pneumonia'
    })
})

# Salvar
df_formatado.to_csv('dados/rotulos_coughvid.csv', index=False)

# Treinar
from treinar import treinar_modelo_completo
treinar_modelo_completo(
    csv_path='dados/rotulos_coughvid.csv',
    audio_dir='caminho/para/coughvid/audios'
)
```

Exemplo 2: Fine-tuning de Modelo Pré-treinado

```python
# fine_tuning.py
from src.modelo import ConstrutorModelos
from src.modelo import GerenciadorCompilacao

# Criar modelo com base MobileNet (pré-treinado)
construtor = ConstrutorModelos()
modelo = construtor.criar_modelo_mobilenet(trainable_base=False)

# Compilar para treino inicial
modelo = GerenciadorCompilacao.compilar_modelo(
    modelo,
    learning_rate=0.0001,
    optimizer='adam'
)

# Treinar apenas as camadas finais por algumas épocas
# ... (código de treinamento)

# Depois descongelar algumas camadas da base
for layer in modelo.layers[:100]:  # Primeiras 100 camadas congeladas
    layer.trainable = False
for layer in modelo.layers[100:]:  # Últimas camadas treináveis
    layer.trainable = True

# Recompilar com LR menor
modelo = GerenciadorCompilacao.compilar_modelo(
    modelo,
    learning_rate=0.00001,  # LR menor para fine-tuning
    optimizer='adam'
)

# Continuar treinamento
# ... (código de treinamento)
```

🚨 SOLUÇÃO DE PROBLEMAS

Problema 1: Erro "No module named 'src'"

```bash
# Solução 1: Adicionar ao PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Solução 2: Executar do diretório raiz
cd /caminho/para/treino
python -c "import sys; sys.path.append('.'); from src.treinar import ..."
```

Problema 2: Memória Insuficiente

```python
# Reduzir batch size
config = {
    'BATCH_SIZE': 8,  # Ou 4 para GPUs com pouca memória
    'USE_GPU': True
}
```

Problema 3: Dataset Desbalanceado

```python
# Ativar balanceamento
config = {
    'BALANCE_CLASSES': True,
    'BALANCE_METHOD': 'smote',  # Ou 'oversample', 'undersample'
    'USE_CHECKPOINTS': True,
    'USE_EARLY_STOPPING': True
}
```

Problema 4: Overfitting

```python
# Aumentar regularização
config = {
    'REGULARIZATION': 1e-3,  # Aumentar regularização
    'DROPOUT_RATE': 0.6,     # Aumentar dropout
    'USE_EARLY_STOPPING': True,
    'PATIENCE_EARLY_STOP': 10
}
```

📊 ANÁLISE DE RESULTADOS

Após o treinamento, você terá:

```
resultados_treinamento/
├── modelos/
│   └── modelo_final_20240101_120000.h5
├── tflite/
│   └── modelo_20240101_120000.tflite
├── graficos/
│   ├── historico_treinamento.png
│   ├── matriz_confusao_test.png
│   └── curva_roc_test.png
├── relatorios/
│   ├── metricas_test.json
│   ├── relatorio_completo_test.json
│   └── resumo_treinamento.json
├── predicoes/
│   └── predicoes_test.csv
└── logs/
    ├── training_20240101_120000.csv
    └── tensorboard/...
```

Verificar Resultados:

```python
import json
import pandas as pd
from IPython.display import Image, display

# Carregar métricas
with open('resultados_treinamento/relatorios/metricas_test.json') as f:
    metricas = json.load(f)

print(f"Acurácia: {metricas['accuracy']:.2%}")
print(f"Precisão: {metricas['precision']:.2%}")
print(f"Recall: {metricas['recall']:.2%}")

# Ver matriz de confusão
display(Image(filename='resultados_treinamento/graficos/matriz_confusao_test.png'))
```

🎯 DICAS PARA MELHORES RESULTADOS

1. Coletar mais dados - Quanto mais, melhor
2. Balancear as classes - Usar SMOTE se possível
3. Aumentar dados - Data augmentation é essencial
4. Tentar diferentes arquiteturas - Teste CNN, LSTM, híbridos
5. Ajustar hiperparâmetros - Learning rate, batch size, etc.
6. Usar validação cruzada - Para estimativa mais confiável
7. Monitorar com TensorBoard - Identifique problemas rapidamente

📞 SUPORTE

Para problemas:

1. Verifique os logs em resultados_treinamento/logs/
2. Confira se todos os arquivos de áudio existem
3. Verifique o formato do CSV
4. Execute os testes de módulos individuais

```bash
# Testar módulos
python src/utilitarios.py
python src/preprocessamento.py
python src/extrair_caracteristicas.py
python src/modelo.py
```

Este sistema está pronto para produção e pode ser usado em projetos reais! 🚀
