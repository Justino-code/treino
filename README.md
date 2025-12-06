📱 Projeto: Classificação de Sons de Tosse para Identificação de Pneumonia e Bronquite

📋 Descrição

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de analisar sons de tosse e auxiliar na identificação de possíveis casos de pneumonia e bronquite. O modelo é projetado para ser integrado a uma aplicação móvel, utilizando processamento local para garantir privacidade e acessibilidade.

🎯 Objetivo

Construir um classificador de áudio que diferencie entre:

· Tosse normal
· Tosse associada à bronquite
· Tosse associada à pneumonia

🏗️ Estrutura do Projeto

```
treino/
├── dados/
│   ├── brutos/           # Áudios originais em formato WAV/MP3
│   ├── processados/      # Características extraídas (espectrogramas, MFCCs)
│   └── rotulos.csv       # Mapeamento: caminho_audio → rótulo (0,1,2)
├── notebooks/
│   └── exploracao.ipynb  # Análise exploratória e visualização dos dados
├── src/
│   ├── __init__.py       # Inicialização do módulo
│   ├── preprocessamento.py       # Funções para processamento de áudio
│   ├── extrair_caracteristicas.py # Extração de features (MFCC, espectrogramas)
│   ├── modelo.py         # Definição das arquiteturas de modelo
│   ├── treinar.py        # Script principal de treinamento
│   └── utilitarios.py    # Funções auxiliares
├── modelos_salvos/       # Modelos treinados (.h5, .tflite)
├── logs/                 # Logs do TensorBoard
├── requisitos.txt        # Dependências do projeto
└── README.md            # Este arquivo
```

🔧 Tecnologias Utilizadas

Machine Learning & Deep Learning

· TensorFlow / Keras - Para construção e treinamento dos modelos
· Librosa - Para processamento e extração de features de áudio
· Scikit-learn - Para avaliação e métricas

Processamento de Áudio

· FFmpeg (indiretamente via librosa) - Para conversão de formatos
· SoundFile - Para leitura/escrita de arquivos de áudio

Visualização e Análise

· Matplotlib / Seaborn - Para visualização de dados
· Pandas / NumPy - Para manipulação de dados

📊 Características Extraídas

O sistema extrai as seguintes características dos áudios de tosse:

1. Espectrogramas Mel - Representação visual da frequência ao longo do tempo
2. MFCCs (Mel-Frequency Cepstral Coefficients) - 13 coeficientes que capturam o timbre
3. Características adicionais (opcionais):
   · Zero Crossing Rate
   · Energia do sinal
   · Duração da tosse

🧠 Modelos Implementados

1. CNN com Transfer Learning (MobileNetV2)

· Base pré-treinada no ImageNet adaptada para espectrogramas
· Adequado para dispositivos móveis após conversão para TFLite

2. CNN Personalizada

· Arquitetura mais leve e específica para áudio
· Menor consumo de recursos computacionais

🚀 Fluxo de Trabalho

1. Preparação dos Dados

· Coleta de áudios de tosse de datasets públicos
· Rotulagem manual/automática (normal, bronquite, pneumonia)
· Padronização: 16kHz, mono, 16-bit

2. Pré-processamento

· Remoção de ruído (filtro de Wiener)
· Normalização de amplitude
· Segmentação (se necessário)

3. Extração de Features

· Conversão para espectrogramas Mel (128x128)
· Extração de MFCCs
· Salvamento em formato numpy para treinamento rápido

4. Treinamento do Modelo

· Divisão dos dados (70% treino, 15% validação, 15% teste)
· Data augmentation (pitch shift, noise injection)
· Treinamento com early stopping e redução de learning rate

5. Otimização para Mobile

· Conversão para TensorFlow Lite
· Quantização (INT8 para eficiência máxima)
· Teste em dispositivos Android/iOS

📁 Datasets Recomendados

1. COUGHVID (EPFL) - Sons de tosse com sintomas respiratórios
2. ICBHI Respiratory Sound Database - Clássico para doenças pulmonares
3. Coswara Dataset - Sons de tosse, respiração e fala
4. Healthily Cough Dataset - Tosse classificada por especialistas

⚠️ Avisos Importantes

· ⚕️ NÃO É UM DIAGNÓSTICO MÉDICO - Este sistema é apenas uma ferramenta de auxílio
· 🔒 Privacidade - Todo processamento pode ser feito localmente no dispositivo
· 📱 Limitações - Eficácia pode variar com qualidade do áudio e ruído ambiental
· 🧪 Validação - Necessária validação com profissionais de saúde

📈 Métricas de Avaliação

· Acurácia geral
· Precisão, Recall e F1-Score por classe
· Matriz de confusão
· Curvas ROC (se aplicável)

🔄 Próximos Passos

1. Coletar mais dados de diferentes fontes
2. Implementar data augmentation avançada para áudio
3. Testar arquiteturas Transformer para áudio (AST)
4. Desenvolver API para análise complementar
5. Integrar com aplicativo móvel (Flutter/React Native)

📝 Como Contribuir

1. Clone o repositório
2. Instale as dependências: pip install -r requisitos.txt
3. Adicione seus áudios na pasta dados/brutos/
4. Atualize dados/rotulos.csv com os caminhos e rótulos
5. Execute o script de treinamento: python src/treinar.py

📄 Licença

Este projeto é destinado a fins educacionais e de pesquisa. O uso para diagnóstico médico real requer aprovação de órgãos regulatórios e validação clínica.

👥 Contato

Para dúvidas ou colaborações, entre em contato com a equipe do projeto.

---

Nota: Este projeto faz parte de uma iniciativa acadêmica para explorar o potencial da IA na área da saúde respiratória.
