# 🤖 Sistema de Classificação de Tosse

Sistema completo para classificação automática de sons de tosse em:
- 🟢 **Normal** - Tosse saudável
- 🟡 **Bronquite** - Tosse com bronquite
- 🔴 **Pneumonia** - Tosse com pneumonia

## 🚀 COMEÇAR RÁPIDO

### 1. Instalação
```bash
# Clone o repositório ou copie os arquivos
# Instale dependências
pip install -r requisitos.txt
```

2. Preparar Dados

```bash
# Crie a estrutura
make setup

# OU
python main.py --status  # Verifica estrutura
```

3. Adicionar Áudios

Coloque seus áudios nas pastas:

```
dados/brutos/normal/*.wav
dados/brutos/bronquite/*.wav  
dados/brutos/pneumonia/*.wav
```

4. Treinar Modelo

```bash
# Usando Python
python main.py --treinar

# OU usando Make
make train
```

5. Usar no Celular

```bash
python main.py --tflite
# O arquivo modelos/modelo_tosse.tflite está pronto!
```

📋 COMANDOS PRINCIPAIS

Comandos Python

```bash
# Treinar novo modelo
python main.py --treinar --epochs 100 --batch 32

# Melhorar modelo existente
python main.py --melhorar

# Converter para celular
python main.py --tflite

# Testar com novos áudios
python main.py --testar --audio-dir meus_audios/

# Verificar status
python main.py --status

# Limpar modelos antigos
python main.py --limpar

# Interface web
python main.py --web
```

Comandos Make (Linux/Mac)

```bash
# Instalar dependências
make install

# Treinar
make train EPOCHS=100 BATCH_SIZE=32

# Converter para TFLite
make tflite

# Verificar status
make status

# Limpar
make clean KEEP_MODELS=5
```

📁 ESTRUTURA DO PROJETO

```
sistema_tosse/
├── main.py              # 🚀 SISTEMA PRINCIPAL
├── Makefile             # Comandos make (Linux/Mac)
├── requisitos.txt       # Dependências
├── dados/
│   └── brutos/
│       ├── normal/      # 🎵 Coloque áudios aqui
│       ├── bronquite/   # 🎵 Coloque áudios aqui  
│       └── pneumonia/   # 🎵 Coloque áudios aqui
├── src/                 # Código fonte
├── modelos/             # Modelos treinados
│   ├── checkpoints/     # Melhores durante treino
│   ├── tflite/          # Modelos para celular
│   └── exportados/      # Outros formatos
├── resultados/          # Resultados e relatórios
├── logs/                # Logs de execução
└── testes/              # Testes e validação
```

🔧 OPÇÕES AVANÇADAS

Tipos de Modelo

```bash
# CNN básica (rápida, boa para começar)
python main.py --treinar --modelo cnn_basico

# CNN avançada (melhor precisão)
python main.py --treinar --modelo cnn_avancado

# MobileNet (leve para celular)
python main.py --treinar --modelo mobilenet
```

Fine-tuning

```bash
# Melhorar modelo existente com mais dados
python main.py --melhorar --epochs 30
```

Testes

```bash
# Testar com diretório específico
python main.py --testar --audio-dir testes/audios/

# Ver resultados detalhados
cat testes/resultados/teste.json | python -m json.tool
```

📊 MONITORAMENTO

Durante Treinamento

· Checkpoints: modelos/checkpoints/melhor_modelo.h5 (salvo automaticamente)
· Logs: logs/treinamento_*.log
· TensorBoard: logs/tensorboard/ (se habilitado)

Após Treinamento

· Modelo final: modelos/modelo_final_*.h5
· TFLite: modelos/tflite/modelo_tosse.tflite
· Gráficos: resultados/graficos/
· Relatórios: resultados/relatorios/

🆘 SOLUÇÃO DE PROBLEMAS

❌ "Nenhum áudio encontrado"

```bash
# Verifique estrutura
python main.py --status

# Crie pastas se necessário
mkdir -p dados/brutos/{normal,bronquite,pneumonia}
```

❌ "ModuleNotFoundError"

```bash
# Instale dependências
pip install -r requisitos.txt
```

❌ "Memória insuficiente"

```bash
# Reduza batch size
python main.py --treinar --batch 8

# Use modelo mais leve
python main.py --treinar --modelo cnn_basico
```

❌ "GPU não detectada"

```bash
# O sistema usará CPU automaticamente
# Para melhor performance com GPU, instale:
pip install tensorflow[and-cuda]
```

📱 USO NO CELULAR

Após converter para TFLite:

1. Copie modelos/tflite/modelo_tosse.tflite para seu projeto Android/iOS
2. Veja modelos/tflite/exemplo_uso.py para código de exemplo
3. Para Flutter, use o pacote tflite_flutter

📈 DICAS PARA MELHORES RESULTADOS

1. Quantidade de dados: Mínimo 50 áudios por classe
2. Qualidade: Áudios claros, sem ruído excessivo
3. Formato: Prefira .wav em 16kHz
4. Duração: 1-5 segundos por áudio
5. Validação: Teste com dados variados

📞 SUPORTE

1. Verificar status: python main.py --status
2. Ver logs: tail -f logs/*.log
3. Testar rapidamente: make dev

---

Desenvolvido para auxiliar na identificação precoce de problemas respiratórios
⚠️AVISO: Este sistema é uma ferramenta de auxílio, não substitui diagnóstico médico.

```

## **🎯 RESUMO FINAL: COMO USAR**

### **PARA USUÁRIOS FINAIS:**

```bash
# SEQUÊNCIA COMPLETA SIMPLES:

# 1. Instalar
pip install -r requisitos.txt

# 2. Preparar (cria pastas)
python main.py --status

# 3. Colocar áudios em:
#    dados/brutos/normal/*.wav
#    dados/brutos/bronquite/*.wav
#    dados/brutos/pneumonia/*.wav

# 4. Treinar
python main.py --treinar

# 5. Usar no celular
python main.py --tflite
```

PARA DESENVOLVEDORES (Linux/Mac):

```bash
# Usando Makefile (mais simples)

# Configurar tudo
make setup

# Treinar com parâmetros
make train EPOCHS=100 BATCH_SIZE=32

# Ciclo completo de desenvolvimento
make dev

# Converter para produção
make tflite
```

FLUXO COMPLETO DO SALVAMENTO:

```
python main.py --treinar
│
├── 📁 DURANTE treinamento:
│   ├── modelos/checkpoints/melhor_modelo.h5  (atualizado quando melhora)
│   └── logs/treinamento_*.log
│
├── ✅ NO FINAL do treinamento:
│   ├── modelos/modelo_final_[DATA].h5        (modelo completo)
│   ├── resultados/graficos/*.png             (gráficos)
│   └── resultados/relatorios/*.json          (métricas)
│
└── 🚀 APÓS treinamento:
    └── python main.py --tflite
        └── modelos/tflite/modelo_tosse.tflite  (para celular)
```

COMANDOS MAIS ÚTEIS:

```bash
# 🚀 Treinar (sempre salva no final)
python main.py --treinar

# 🔧 Melhorar modelo existente
python main.py --melhorar

# 📱 Preparar para celular
python main.py --tflite

# 🧪 Testar com novos áudios
python main.py --testar

# 📊 Verificar tudo
python main.py --status

# 🧹 Manter organizado
python main.py --limpar
```

Agora você tem um sistema completo e profissional com:

1. main.py - Interface CLI poderosa
2. Makefile - Para usuários Linux/Mac
3. Salvamento automático durante e após treinamento
4. Múltiplas opções de exportação
5. Interface web simples
6. Gestão completa de modelos

Basta executar python main.py --treinar e o sistema cuida de tudo! 🚀
