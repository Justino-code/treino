# Makefile - Comandos para Sistema de Classificação de Tosse

.PHONY: help install train improve tflite test status clean web

# Cores para output
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m # No Color

# Configurações padrão
EPOCHS = 50
BATCH_SIZE = 16
MODEL_TYPE = cnn_basico
KEEP_MODELS = 3

help:
	@echo "$(GREEN)Sistema de Classificação de Tosse$(NC)"
	@echo ""
	@echo "$(YELLOW)Comandos disponíveis:$(NC)"
	@echo ""
	@echo "  $(GREEN)install$(NC)     - Instala dependências"
	@echo "  $(GREEN)train$(NC)       - Treina novo modelo (make train EPOCHS=100)"
	@echo "  $(GREEN)improve$(NC)     - Melhora modelo existente"
	@echo "  $(GREEN)tflite$(NC)      - Converte para TensorFlow Lite"
	@echo "  $(GREEN)test$(NC)        - Testa modelo com novos áudios"
	@echo "  $(GREEN)status$(NC)      - Mostra status do sistema"
	@echo "  $(GREEN)clean$(NC)       - Limpa modelos antigos"
	@echo "  $(GREEN)web$(NC)         - Inicia interface web"
	@echo "  $(GREEN)check$(NC)       - Verifica dados e estrutura"
	@echo ""
	@echo "$(YELLOW)Exemplos:$(NC)"
	@echo "  make train EPOCHS=100 BATCH_SIZE=32"
	@echo "  make tflite"
	@echo "  make test AUDIO_DIR=meus_audios/"

install:
	@echo "$(GREEN)Instalando dependências...$(NC)"
	pip install -r requisitos.txt
	@echo "$(GREEN)✅ Dependências instaladas$(NC)"

check:
	@echo "$(GREEN)Verificando sistema...$(NC)"
	python main.py --status

train:
	@echo "$(GREEN)Treinando modelo...$(NC)"
	python main.py --treinar --epochs $(EPOCHS) --batch $(BATCH_SIZE) --modelo $(MODEL_TYPE)

improve:
	@echo "$(GREEN)Melhorando modelo...$(NC)"
	python main.py --melhorar --epochs $(EPOCHS)

tflite:
	@echo "$(GREEN)Convertendo para TFLite...$(NC)"
	python main.py --tflite

test:
	@echo "$(GREEN)Testando modelo...$(NC)"
ifdef AUDIO_DIR
	python main.py --testar --audio-dir $(AUDIO_DIR)
else
	python main.py --testar
endif

status:
	@echo "$(GREEN)Status do sistema:$(NC)"
	python main.py --status

clean:
	@echo "$(YELLOW)Limpando modelos antigos...$(NC)"
	python main.py --limpar --manter $(KEEP_MODELS)

web:
	@echo "$(GREEN)Iniciando interface web...$(NC)"
	python main.py --web

setup:
	@echo "$(GREEN)Configurando sistema...$(NC)"
	mkdir -p dados/brutos/{normal,bronquite,pneumonia}
	mkdir -p modelos/{checkpoints,tflite}
	mkdir -p resultados/{graficos,relatorios,predicoes}
	mkdir -p testes/{audios,resultados}
	mkdir -p logs
	@echo "$(GREEN)✅ Estrutura criada$(NC)"
	@echo ""
	@echo "$(YELLOW)Próximos passos:$(NC)"
	@echo "1. Coloque áudios em dados/brutos/normal/, bronquite/, pneumonia/"
	@echo "2. Execute: make train"
	@echo "3. Para usar no celular: make tflite"

# Comando especial: treinamento completo
all: install setup train tflite
	@echo "$(GREEN)✅ Pipeline completo executado!$(NC)"

# Comando para desenvolvimento
dev: install check train test
	@echo "$(GREEN)✅ Ciclo de desenvolvimento completo$(NC)"

# Backup dos modelos
backup:
	@echo "$(GREEN)Criando backup dos modelos...$(NC)"
	tar -czf modelos_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz modelos/
	@echo "$(GREEN)✅ Backup criado$(NC)"

# Listar modelos
list:
	@echo "$(GREEN)Modelos disponíveis:$(NC)"
	@find modelos/ -name "*.h5" -o -name "*.tflite" | sort -r
