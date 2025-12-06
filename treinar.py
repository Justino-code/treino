"""
treinar.py - Módulo principal para treinamento de modelos de classificação de tosse

Este módulo contém:
- Pipeline completo de treinamento
- Validação cruzada
- Balanceamento de dados
- Data augmentation
- Avaliação de modelos
- Geração de relatórios
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
import pickle
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from pathlib import Path
import shutil

# Importar módulos personalizados
from . import utilitarios
from . import preprocessamento
from . import extrair_caracteristicas
from . import modelo

# Configurar logging
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAÇÕES DE TREINAMENTO
# ============================================================================

class ConfigTreinamento:
    """Configurações padrão para treinamento."""
    
    # Configurações gerais
    SEED = 42
    VERBOSE = 1
    USE_GPU = True
    
    # Configurações de dados
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    CV_FOLDS = 5
    
    # Data augmentation
    AUGMENT_TRAIN = True
    AUGMENT_VAL = False
    AUGMENT_PROB = 0.5
    
    # Balanceamento
    BALANCE_CLASSES = True
    BALANCE_METHOD = 'smote'  # 'smote', 'undersample', 'oversample'
    
    # Configurações de modelo
    MODEL_TYPE = 'cnn_avancado'  # 'cnn_basico', 'cnn_avancado', 'mobilenet', 'efficientnet'
    USE_PRETRAINED = False
    
    # Configurações de otimização
    LEARNING_RATE = 1e-4
    OPTIMIZER = 'adam'
    LOSS = 'categorical_crossentropy'
    METRICS = ['accuracy', 'precision', 'recall', 'auc']
    
    # Configurações de treinamento
    BATCH_SIZE = 32
    EPOCHS = 100
    INITIAL_EPOCH = 0
    
    # Configurações de callbacks
    USE_CHECKPOINTS = True
    USE_EARLY_STOPPING = True
    USE_REDUCE_LR = True
    USE_TENSORBOARD = True
    
    # Configurações de avaliação
    EVALUATE_ON_TEST = True
    GENERATE_REPORTS = True
    SAVE_PREDICTIONS = True
    CONFIDENCE_THRESHOLD = 0.5
    
    # Configurações de arquivos
    MODEL_SAVE_DIR = 'modelos_salvos'
    LOGS_DIR = 'logs'
    RESULTS_DIR = 'resultados'
    CHECKPOINT_DIR = 'checkpoints'

# ============================================================================
# CLASSE PRINCIPAL DE TREINAMENTO
# ============================================================================

class TreinadorTosse:
    """
    Classe principal para treinamento de modelos de classificação de tosse.
    
    Gerencia todo o pipeline: carregamento de dados, pré-processamento,
    treinamento, avaliação e geração de relatórios.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        data_dir: str = 'dados',
        resultados_dir: str = 'resultados'
    ):
        """
        Inicializa o treinador.
        
        Args:
            config: Dicionário com configurações (None para usar padrões)
            data_dir: Diretório com dados
            resultados_dir: Diretório para salvar resultados
        """
        # Configurações
        self.config = self._carregar_config(config)
        
        # Diretórios
        self.data_dir = Path(data_dir)
        self.resultados_dir = Path(resultados_dir)
        
        # Criar diretórios necessários
        self._criar_diretorios()
        
        # Componentes
        self.preprocessador = None
        self.extrator = None
        self.construtor_modelos = None
        self.model = None
        self.history = None
        
        # Dados
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.class_names = None
        
        # Estatísticas
        self.class_distribution = None
        self.training_stats = {}
        
        # Configurar GPU se disponível
        self._configurar_gpu()
        
        logger.info("Treinador de tosse inicializado")
    
    def _carregar_config(self, config: Optional[Dict]) -> Dict:
        """Carrega configurações, mesclando com padrões."""
        config_padrao = {
            k: v for k, v in ConfigTreinamento.__dict__.items() 
            if not k.startswith('_')
        }
        
        if config:
            config_padrao.update(config)
        
        return config_padrao
    
    def _criar_diretorios(self):
        """Cria todos os diretórios necessários."""
        dirs = [
            self.resultados_dir / 'modelos',
            self.resultados_dir / 'checkpoints',
            self.resultados_dir / 'logs',
            self.resultados_dir / 'graficos',
            self.resultados_dir / 'relatorios',
            self.resultados_dir / 'predicoes',
            self.resultados_dir / 'tflite'
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Diretório criado/verificado: {dir_path}")
    
    def _configurar_gpu(self):
        """Configura uso de GPU se disponível."""
        if self.config['USE_GPU']:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Permitir crescimento de memória
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    logger.info(f"GPU detectada: {len(gpus)} dispositivos")
                    
                    # Configurar estratégia de distribuição se múltiplas GPUs
                    if len(gpus) > 1:
                        strategy = tf.distribute.MirroredStrategy()
                        logger.info(f"Usando {len(gpus)} GPUs com MirroredStrategy")
                        return strategy
                    else:
                        logger.info("Usando GPU única")
                        
                except Exception as e:
                    logger.warning(f"Erro ao configurar GPU: {e}. Usando CPU.")
            else:
                logger.info("Nenhuma GPU detectada. Usando CPU.")
        else:
            logger.info("GPU desativada por configuração. Usando CPU.")
        
        return None
    
    # ============================================================================
    # CARREGAMENTO E PREPARAÇÃO DE DADOS
    # ============================================================================
    
    def carregar_dados(
        self,
        csv_path: str,
        audio_dir: str = None,
        classes: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> bool:
        """
        Carrega dados de um arquivo CSV.
        
        Args:
            csv_path: Caminho para arquivo CSV com rótulos
            audio_dir: Diretório com áudios (None para usar caminhos absolutos)
            classes: Lista de classes (None para inferir dos dados)
            limit: Limitar número de amostras (None para todos)
        
        Returns:
            True se carregado com sucesso, False caso contrário
        """
        logger.info(f"Carregando dados de {csv_path}")
        
        try:
            # Ler CSV
            df = pd.read_csv(csv_path)
            
            # Limitar amostras se especificado
            if limit and len(df) > limit:
                df = df.sample(limit, random_state=self.config['SEED'])
                logger.info(f"Limitado a {limit} amostras")
            
            # Processar caminhos de áudio
            if audio_dir:
                audio_dir_path = Path(audio_dir)
                df['caminho_completo'] = df['caminho'].apply(
                    lambda x: str(audio_dir_path / x) if pd.notnull(x) else None
                )
            else:
                df['caminho_completo'] = df['caminho']
            
            # Remover entradas inválidas
            df = df.dropna(subset=['caminho_completo', 'rotulo'])
            
            # Verificar se arquivos existem
            existentes = df['caminho_completo'].apply(lambda x: Path(x).exists())
            n_existentes = existentes.sum()
            
            if n_existentes < len(df):
                logger.warning(f"{len(df) - n_existentes} arquivos não encontrados")
                df = df[existentes]
            
            if len(df) == 0:
                logger.error("Nenhum dado válido encontrado")
                return False
            
            # Mapear rótulos para classes
            if classes:
                self.class_names = classes
            else:
                self.class_names = sorted(df['rotulo'].unique().tolist())
            
            label_to_index = {label: i for i, label in enumerate(self.class_names)}
            df['label_index'] = df['rotulo'].map(label_to_index)
            
            # Estatísticas das classes
            self.class_distribution = df['rotulo'].value_counts().to_dict()
            logger.info(f"Distribuição de classes: {self.class_distribution}")
            
            # Salvar caminhos e rótulos
            self.data_paths = df['caminho_completo'].tolist()
            self.data_labels = df['label_index'].tolist()
            
            logger.info(f"Dados carregados: {len(self.data_paths)} amostras, "
                       f"{len(self.class_names)} classes")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            return False
    
    def preparar_dados(
        self,
        test_size: float = None,
        val_size: float = None,
        augment: bool = None
    ):
        """
        Prepara dados para treinamento: pré-processa, divide e aumenta.
        
        Args:
            test_size: Proporção para teste (None para usar config)
            val_size: Proporção para validação (None para usar config)
            augment: Se True, aplica data augmentation (None para usar config)
        """
        logger.info("Preparando dados para treinamento...")
        
        # Usar configurações se não especificado
        if test_size is None:
            test_size = self.config['TEST_SIZE']
        if val_size is None:
            val_size = self.config['VAL_SIZE']
        if augment is None:
            augment = self.config['AUGMENT_TRAIN']
        
        try:
            # Inicializar componentes se necessário
            if self.preprocessador is None:
                self.preprocessador = preprocessamento.PreprocessadorTosse()
            
            if self.extrator is None:
                self.extrator = extrair_caracteristicas.ExtratorCaracteristicasTosse()
            
            # 1. Pré-processar áudios
            logger.info("Pré-processando áudios...")
            audios_processados = self.preprocessador.processar_lote(
                self.data_paths,
                usar_multithreading=True
            )
            
            # 2. Extrair características (como imagens)
            logger.info("Extraindo características...")
            caracteristicas = []
            
            for i, audio in enumerate(audios_processados):
                try:
                    # Extrair espectrograma e converter para imagem
                    mel_spec = self.extrator.extrair_espectrograma_mel(audio)
                    imagem = self.extrator.espectrograma_para_imagem(mel_spec)
                    caracteristicas.append(imagem)
                except Exception as e:
                    logger.warning(f"Erro ao extrair características da amostra {i}: {e}")
                    # Adicionar zeros como fallback
                    caracteristicas.append(np.zeros((128, 128, 3)))
                
                # Progresso
                if (i + 1) % 100 == 0:
                    logger.info(f"Extraídas características de {i+1}/{len(audios_processados)} amostras")
            
            X = np.array(caracteristicas)
            y = np.array(self.data_labels)
            
            logger.info(f"Dados preparados: X.shape={X.shape}, y.shape={y.shape}")
            
            # 3. Dividir dados
            logger.info("Dividindo dados...")
            
            # Primeiro separar teste
            from sklearn.model_selection import train_test_split
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                X, y,
                test_size=test_size,
                stratify=y,
                random_state=self.config['SEED']
            )
            
            # Depois separar validação do restante
            val_relative = val_size / (1 - test_size)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_relative,
                stratify=y_temp,
                random_state=self.config['SEED']
            )
            
            logger.info(f"Divisão: "
                       f"Treino={len(self.X_train)}, "
                       f"Validação={len(self.X_val)}, "
                       f"Teste={len(self.X_test)}")
            
            # 4. Balancear classes se necessário
            if self.config['BALANCE_CLASSES']:
                self._balancear_dados()
            
            # 5. Aplicar data augmentation se necessário
            if augment:
                self._aplicar_data_augmentation()
            
            # 6. Converter rótulos para one-hot encoding
            self.y_train = keras.utils.to_categorical(self.y_train, len(self.class_names))
            self.y_val = keras.utils.to_categorical(self.y_val, len(self.class_names))
            self.y_test_cat = keras.utils.to_categorical(self.y_test, len(self.class_names))
            
            # 7. Normalizar dados (se necessário)
            self._normalizar_dados()
            
            # 8. Criar datasets TensorFlow
            self._criar_datasets()
            
            logger.info("Dados preparados com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados: {e}")
            raise
    
    def _balancear_dados(self):
        """Aplica técnicas de balanceamento de classes."""
        logger.info("Balanceando classes...")
        
        try:
            method = self.config['BALANCE_METHOD']
            
            if method == 'undersample':
                self._balancear_undersample()
            elif method == 'oversample':
                self._balancear_oversample()
            elif method == 'smote':
                self._balancear_smote()
            else:
                logger.warning(f"Método de balanceamento desconhecido: {method}")
                
        except Exception as e:
            logger.error(f"Erro no balanceamento: {e}. Continuando sem balanceamento.")
    
    def _balancear_undersample(self):
        """Aplica undersampling para balancear classes."""
        from sklearn.utils import resample
        
        # Combinar X e y para facilitar
        train_data = list(zip(self.X_train, self.y_train))
        
        # Separar por classe
        classes = np.unique(self.y_train)
        min_samples = min([np.sum(self.y_train == c) for c in classes])
        
        balanced_data = []
        for c in classes:
            class_data = [d for d in train_data if d[1] == c]
            undersampled = resample(class_data, 
                                   replace=False, 
                                   n_samples=min_samples, 
                                   random_state=self.config['SEED'])
            balanced_data.extend(undersampled)
        
        # Separar novamente
        self.X_train = np.array([d[0] for d in balanced_data])
        self.y_train = np.array([d[1] for d in balanced_data])
        
        logger.info(f"Após undersampling: {len(self.X_train)} amostras")
    
    def _balancear_oversample(self):
        """Aplica oversampling para balancear classes."""
        from sklearn.utils import resample
        
        # Combinar X e y para facilitar
        train_data = list(zip(self.X_train, self.y_train))
        
        # Separar por classe
        classes = np.unique(self.y_train)
        max_samples = max([np.sum(self.y_train == c) for c in classes])
        
        balanced_data = []
        for c in classes:
            class_data = [d for d in train_data if d[1] == c]
            n_samples = len(class_data)
            
            if n_samples < max_samples:
                oversampled = resample(class_data, 
                                      replace=True, 
                                      n_samples=max_samples, 
                                      random_state=self.config['SEED'])
                balanced_data.extend(oversampled)
            else:
                balanced_data.extend(class_data)
        
        # Separar novamente
        self.X_train = np.array([d[0] for d in balanced_data])
        self.y_train = np.array([d[1] for d in balanced_data])
        
        logger.info(f"Após oversampling: {len(self.X_train)} amostras")
    
    def _balancear_smote(self):
        """Aplica SMOTE para balancear classes."""
        try:
            from imblearn.over_sampling import SMOTE
            
            # SMOTE requer dados 2D, então achatar imagens temporariamente
            original_shape = self.X_train.shape
            X_flat = self.X_train.reshape(len(self.X_train), -1)
            
            smote = SMOTE(random_state=self.config['SEED'])
            X_balanced, y_balanced = smote.fit_resample(X_flat, self.y_train)
            
            # Restaurar shape original
            self.X_train = X_balanced.reshape(-1, *original_shape[1:])
            self.y_train = y_balanced
            
            logger.info(f"Após SMOTE: {len(self.X_train)} amostras")
            
        except ImportError:
            logger.warning("imblearn não instalado. Usando oversampling simples.")
            self._balancear_oversample()
    
    def _aplicar_data_augmentation(self):
        """Aplica data augmentation aos dados de treino."""
        logger.info("Aplicando data augmentation...")
        
        try:
            # Criar gerador de aumento de dados
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                brightness_range=[0.9, 1.1],
                fill_mode='nearest'
            )
            
            # Aumentar dados
            n_augmented = int(len(self.X_train) * self.config['AUGMENT_PROB'])
            
            if n_augmented > 0:
                # Gerar dados aumentados
                X_augmented = []
                y_augmented = []
                
                datagen.fit(self.X_train)
                
                for X_batch, y_batch in datagen.flow(
                    self.X_train, 
                    self.y_train,
                    batch_size=n_augmented,
                    shuffle=False
                ):
                    X_augmented.append(X_batch)
                    y_augmented.append(y_batch)
                    break  # Apenas um batch
                
                # Combinar com dados originais
                self.X_train = np.concatenate([self.X_train, X_augmented[0]], axis=0)
                self.y_train = np.concatenate([self.y_train, y_augmented[0]], axis=0)
                
                logger.info(f"Após augmentation: {len(self.X_train)} amostras")
                
        except Exception as e:
            logger.error(f"Erro no data augmentation: {e}")
    
    def _normalizar_dados(self):
        """Normaliza os dados de entrada."""
        logger.info("Normalizando dados...")
        
        try:
            # Calcular estatísticas de normalização
            self.mean = np.mean(self.X_train, axis=(0, 1, 2, 3), keepdims=True)
            self.std = np.std(self.X_train, axis=(0, 1, 2, 3), keepdims=True) + 1e-7
            
            # Aplicar normalização
            self.X_train = (self.X_train - self.mean) / self.std
            self.X_val = (self.X_val - self.mean) / self.std
            self.X_test = (self.X_test - self.mean) / self.std
            
            logger.info(f"Normalização aplicada: mean={self.mean.mean():.3f}, std={self.std.mean():.3f}")
            
        except Exception as e:
            logger.error(f"Erro na normalização: {e}")
    
    def _criar_datasets(self):
        """Cria datasets TensorFlow para treinamento eficiente."""
        logger.info("Criando datasets TensorFlow...")
        
        try:
            # Dataset de treino
            self.train_dataset = tf.data.Dataset.from_tensor_slices(
                (self.X_train, self.y_train)
            ).shuffle(
                buffer_size=len(self.X_train),
                reshuffle_each_iteration=True,
                seed=self.config['SEED']
            ).batch(
                self.config['BATCH_SIZE']
            ).prefetch(
                tf.data.AUTOTUNE
            )
            
            # Dataset de validação
            self.val_dataset = tf.data.Dataset.from_tensor_slices(
                (self.X_val, self.y_val)
            ).batch(
                self.config['BATCH_SIZE']
            ).prefetch(
                tf.data.AUTOTUNE
            )
            
            # Dataset de teste
            self.test_dataset = tf.data.Dataset.from_tensor_slices(
                (self.X_test, self.y_test_cat)
            ).batch(
                self.config['BATCH_SIZE']
            ).prefetch(
                tf.data.AUTOTUNE
            )
            
            logger.info(f"Datasets criados: "
                       f"Treino={len(self.train_dataset)} batches, "
                       f"Validação={len(self.val_dataset)} batches, "
                       f"Teste={len(self.test_dataset)} batches")
            
        except Exception as e:
            logger.error(f"Erro ao criar datasets: {e}")
    
    # ============================================================================
    # CONSTRUÇÃO E COMPILAÇÃO DE MODELO
    # ============================================================================
    
    def construir_modelo(self, model_type: str = None):
        """
        Constrói e compila o modelo.
        
        Args:
            model_type: Tipo de modelo (None para usar config)
        """
        logger.info("Construindo modelo...")
        
        if model_type is None:
            model_type = self.config['MODEL_TYPE']
        
        try:
            # Inicializar construtor
            self.construtor_modelos = modelo.ConstrutorModelos(
                input_shape=self.X_train.shape[1:],
                num_classes=len(self.class_names)
            )
            
            # Construir modelo baseado no tipo
            if model_type == 'cnn_basico':
                self.model = self.construtor_modelos.criar_modelo_cnn_basico()
            elif model_type == 'cnn_avancado':
                self.model = self.construtor_modelos.criar_modelo_cnn_avancado()
            elif model_type == 'mobilenet':
                self.model = self.construtor_modelos.criar_modelo_mobilenet(
                    trainable_base=self.config['USE_PRETRAINED']
                )
            elif model_type == 'efficientnet':
                self.model = self.construtor_modelos.criar_modelo_eficientnet('B0')
            elif model_type == 'cnn_lstm':
                self.model = self.construtor_modelos.criar_modelo_cnn_lstm_hibrido()
            elif model_type == 'mobile':
                self.model = self.construtor_modelos.criar_modelo_leve_mobile()
            else:
                logger.warning(f"Tipo de modelo desconhecido: {model_type}. Usando CNN básico.")
                self.model = self.construtor_modelos.criar_modelo_cnn_basico()
            
            # Compilar modelo
            gerenciador_compilacao = modelo.GerenciadorCompilacao()
            self.model = gerenciador_compilacao.compilar_modelo(
                self.model,
                learning_rate=self.config['LEARNING_RATE'],
                optimizer=self.config['OPTIMIZER'],
                loss=self.config['LOSS'],
                metrics=self.config['METRICS']
            )
            
            # Mostrar resumo
            logger.info("Resumo do modelo:")
            self.model.summary(print_fn=logger.info)
            
            # Salvar arquitetura do modelo
            arquitetura_path = self.resultados_dir / 'relatorios' / 'arquitetura_modelo.txt'
            with open(arquitetura_path, 'w') as f:
                self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            
            logger.info(f"Arquitetura do modelo salva em: {arquitetura_path}")
            
        except Exception as e:
            logger.error(f"Erro ao construir modelo: {e}")
            raise
    
    def criar_callbacks(self):
        """Cria callbacks para treinamento."""
        logger.info("Criando callbacks...")
        
        try:
            gerenciador_compilacao = modelo.GerenciadorCompilacao()
            
            # Nome base para arquivos
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"modelo_tosse_{timestamp}"
            
            # Configurar diretórios
            checkpoint_dir = self.resultados_dir / 'checkpoints'
            tensorboard_dir = self.resultados_dir / 'logs' / 'tensorboard'
            csv_log_path = self.resultados_dir / 'logs' / f'training_{timestamp}.csv'
            
            # Criar callbacks
            self.callbacks = []
            
            # Checkpoint
            if self.config['USE_CHECKPOINTS']:
                checkpoint_path = checkpoint_dir / f'{model_name}_best.h5'
                checkpoint_callback = keras.callbacks.ModelCheckpoint(
                    filepath=str(checkpoint_path),
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True,
                    verbose=1
                )
                self.callbacks.append(checkpoint_callback)
            
            # Early Stopping
            if self.config['USE_EARLY_STOPPING']:
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.get('PATIENCE_EARLY_STOP', 15),
                    restore_best_weights=True,
                    verbose=1
                )
                self.callbacks.append(early_stopping)
            
            # Reduce LR on Plateau
            if self.config['USE_REDUCE_LR']:
                reduce_lr = keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.config.get('FACTOR_REDUCE_LR', 0.5),
                    patience=self.config.get('PATIENCE_REDUCE_LR', 8),
                    min_lr=self.config.get('MIN_LEARNING_RATE', 1e-7),
                    verbose=1
                )
                self.callbacks.append(reduce_lr)
            
            # TensorBoard
            if self.config['USE_TENSORBOARD']:
                tensorboard = keras.callbacks.TensorBoard(
                    log_dir=str(tensorboard_dir),
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq='epoch'
                )
                self.callbacks.append(tensorboard)
            
            # CSV Logger
            csv_logger = keras.callbacks.CSVLogger(
                str(csv_log_path),
                append=True
            )
            self.callbacks.append(csv_logger)
            
            logger.info(f"Callbacks criados: {len(self.callbacks)} callbacks configurados")
            
        except Exception as e:
            logger.error(f"Erro ao criar callbacks: {e}")
            self.callbacks = []
    
    # ============================================================================
    # TREINAMENTO
    # ============================================================================
    
    def treinar(self, epochs: int = None, initial_epoch: int = None):
        """
        Treina o modelo.
        
        Args:
            epochs: Número de épocas (None para usar config)
            initial_epoch: Época inicial (None para usar config)
        
        Returns:
            Histórico de treinamento
        """
        logger.info("Iniciando treinamento...")
        
        if epochs is None:
            epochs = self.config['EPOCHS']
        if initial_epoch is None:
            initial_epoch = self.config['INITIAL_EPOCH']
        
        try:
            # Verificar se tudo está pronto
            if self.model is None:
                logger.error("Modelo não construído. Chame construir_modelo() primeiro.")
                return None
            
            if self.train_dataset is None:
                logger.error("Dados não preparados. Chame preparar_dados() primeiro.")
                return None
            
            if self.callbacks is None:
                self.criar_callbacks()
            
            # Informações de treinamento
            logger.info(f"Configurações de treinamento:")
            logger.info(f"  Épocas: {epochs}")
            logger.info(f"  Batch size: {self.config['BATCH_SIZE']}")
            logger.info(f"  Amostras de treino: {len(self.X_train)}")
            logger.info(f"  Amostras de validação: {len(self.X_val)}")
            logger.info(f"  Classes: {self.class_names}")
            
            # Iniciar treinamento
            start_time = datetime.now()
            
            self.history = self.model.fit(
                self.train_dataset,
                epochs=epochs,
                initial_epoch=initial_epoch,
                validation_data=self.val_dataset,
                callbacks=self.callbacks,
                verbose=self.config['VERBOSE']
            )
            
            # Calcular tempo de treinamento
            training_time = datetime.now() - start_time
            logger.info(f"Tempo de treinamento: {training_time}")
            
            # Salvar modelo final
            self._salvar_modelo_final()
            
            # Salvar histórico
            self._salvar_historico()
            
            # Atualizar estatísticas
            self.training_stats.update({
                'training_time': str(training_time),
                'final_epoch': len(self.history.history['loss']),
                'final_train_loss': self.history.history['loss'][-1],
                'final_val_loss': self.history.history['val_loss'][-1],
                'final_train_accuracy': self.history.history['accuracy'][-1],
                'final_val_accuracy': self.history.history['val_accuracy'][-1]
            })
            
            logger.info("Treinamento concluído com sucesso")
            
            return self.history
            
        except Exception as e:
            logger.error(f"Erro durante treinamento: {e}")
            raise
    
    def _salvar_modelo_final(self):
        """Salva o modelo treinado."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Salvar modelo Keras
            model_path = self.resultados_dir / 'modelos' / f'modelo_final_{timestamp}.h5'
            self.model.save(str(model_path))
            logger.info(f"Modelo final salvo em: {model_path}")
            
            # Salvar pesos separadamente
            weights_path = self.resultados_dir / 'modelos' / f'pesos_finais_{timestamp}.h5'
            self.model.save_weights(str(weights_path))
            
            # Converter para TensorFlow Lite
            tflite_path = self.resultados_dir / 'tflite' / f'modelo_{timestamp}.tflite'
            gerenciador_modelos = modelo.GerenciadorModelos()
            gerenciador_modelos.converter_para_tflite(
                self.model,
                str(tflite_path),
                quantize=True
            )
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
    
    def _salvar_historico(self):
        """Salva o histórico de treinamento."""
        try:
            history_path = self.resultados_dir / 'relatorios' / 'historico_treinamento.json'
            
            # Converter valores numpy para tipos Python
            history_dict = {}
            for key, values in self.history.history.items():
                history_dict[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                    for v in values]
            
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
            
            logger.info(f"Histórico salvo em: {history_path}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar histórico: {e}")
    
    # ============================================================================
    # AVALIAÇÃO
    # ============================================================================
    
    def avaliar(self, dataset: str = 'test'):
        """
        Avalia o modelo treinado.
        
        Args:
            dataset: Qual dataset avaliar ('test', 'val', 'train')
        
        Returns:
            Dicionário com métricas
        """
        logger.info(f"Avaliando modelo no dataset {dataset}...")
        
        try:
            # Selecionar dataset
            if dataset == 'test':
                X = self.X_test
                y = self.y_test_cat
                y_true = self.y_test
                dataset_obj = self.test_dataset
            elif dataset == 'val':
                X = self.X_val
                y = self.y_val
                y_true = np.argmax(y, axis=1)
                dataset_obj = self.val_dataset
            elif dataset == 'train':
                X = self.X_train
                y = self.y_train
                y_true = np.argmax(y, axis=1)
                dataset_obj = self.train_dataset
            else:
                raise ValueError(f"Dataset desconhecido: {dataset}")
            
            # Avaliar com model.evaluate
            results = self.model.evaluate(dataset_obj, verbose=self.config['VERBOSE'])
            
            # Criar dicionário de resultados
            metrics = {}
            for i, metric_name in enumerate(self.model.metrics_names):
                metrics[metric_name] = float(results[i])
            
            # Fazer predições
            y_pred_prob = self.model.predict(X, verbose=self.config['VERBOSE'])
            y_pred = np.argmax(y_pred_prob, axis=1)
            
            # Calcular métricas adicionais
            from sklearn.metrics import classification_report, confusion_matrix
            
            # Relatório de classificação
            report = classification_report(
                y_true, 
                y_pred, 
                target_names=self.class_names,
                output_dict=True
            )
            metrics['classification_report'] = report
            
            # Matriz de confusão
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Acurácia por classe
            for i, class_name in enumerate(self.class_names):
                class_mask = y_true == i
                if np.any(class_mask):
                    class_accuracy = np.mean(y_pred[class_mask] == i)
                    metrics[f'accuracy_{class_name}'] = float(class_accuracy)
            
            # Salvar métricas
            self._salvar_metricas(metrics, dataset)
            
            # Gerar visualizações
            if self.config['GENERATE_REPORTS']:
                self._gerar_relatorios(y_true, y_pred, y_pred_prob, dataset)
            
            # Salvar predições se solicitado
            if self.config['SAVE_PREDICTIONS']:
                self._salvar_predicoes(X, y_true, y_pred, y_pred_prob, dataset)
            
            logger.info(f"Avaliação concluída para dataset {dataset}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro na avaliação: {e}")
            return {}
    
    def _salvar_metricas(self, metrics: Dict, dataset: str):
        """Salva métricas em arquivo JSON."""
        try:
            metrics_path = self.resultados_dir / 'relatorios' / f'metricas_{dataset}.json'
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            logger.info(f"Métricas salvas em: {metrics_path}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar métricas: {e}")
    
    def _gerar_relatorios(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_prob: np.ndarray,
        dataset: str
    ):
        """Gera relatórios e visualizações."""
        try:
            # Importar utilitários
            from .utilitarios import (
                plotar_historico_treinamento,
                plotar_matriz_confusao,
                plotar_curva_roc,
                calcular_metricas_classificacao,
                gerar_relatorio_treinamento
            )
            
            # 1. Plotar histórico de treinamento
            if self.history:
                history_path = self.resultados_dir / 'graficos' / 'historico_treinamento.png'
                plotar_historico_treinamento(
                    self.history.history,
                    salvar=True,
                    caminho_salvar=str(history_path),
                    mostrar=False
                )
            
            # 2. Plotar matriz de confusão
            cm_path = self.resultados_dir / 'graficos' / f'matriz_confusao_{dataset}.png'
            plotar_matriz_confusao(
                y_true, y_pred,
                classes=self.class_names,
                salvar=True,
                caminho_salvar=str(cm_path),
                mostrar=False
            )
            
            # 3. Plotar curva ROC (se mais de 2 classes)
            if len(self.class_names) > 2:
                roc_path = self.resultados_dir / 'graficos' / f'curva_roc_{dataset}.png'
                plotar_curva_roc(
                    keras.utils.to_categorical(y_true, len(self.class_names)),
                    y_pred_prob,
                    classes=self.class_names,
                    salvar=True,
                    caminho_salvar=str(roc_path),
                    mostrar=False
                )
            
            # 4. Calcular métricas detalhadas
            detailed_metrics = calcular_metricas_classificacao(
                y_true, y_pred, self.class_names
            )
            
            # 5. Gerar relatório completo
            relatorio_path = self.resultados_dir / 'relatorios' / f'relatorio_completo_{dataset}.json'
            gerar_relatorio_treinamento(
                detailed_metrics,
                self.config,
                caminho_salvar=str(relatorio_path)
            )
            
            logger.info(f"Relatórios gerados para dataset {dataset}")
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatórios: {e}")
    
    def _salvar_predicoes(
        self, 
        X: np.ndarray, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_prob: np.ndarray,
        dataset: str
    ):
        """Salva predições em arquivo CSV."""
        try:
            predicoes_df = pd.DataFrame({
                'true_label': [self.class_names[i] for i in y_true],
                'predicted_label': [self.class_names[i] for i in y_pred],
                'correct': y_true == y_pred,
                'confidence': np.max(y_pred_prob, axis=1)
            })
            
            # Adicionar probabilidades por classe
            for i, class_name in enumerate(self.class_names):
                predicoes_df[f'prob_{class_name}'] = y_pred_prob[:, i]
            
            # Salvar
            predicoes_path = self.resultados_dir / 'predicoes' / f'predicoes_{dataset}.csv'
            predicoes_df.to_csv(predicoes_path, index=False)
            
            logger.info(f"Predições salvas em: {predicoes_path}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar predições: {e}")
    
    # ============================================================================
    # VALIDAÇÃO CRUZADA
    # ============================================================================
    
    def validacao_cruzada(
        self, 
        n_folds: int = None,
        n_epochs: int = 50
    ) -> Dict:
        """
        Executa validação cruzada.
        
        Args:
            n_folds: Número de folds (None para usar config)
            n_epochs: Épocas por fold
        
        Returns:
            Dicionário com resultados da validação cruzada
        """
        logger.info(f"Iniciando validação cruzada com {n_folds or self.config['CV_FOLDS']} folds...")
        
        if n_folds is None:
            n_folds = self.config['CV_FOLDS']
        
        try:
            from sklearn.model_selection import StratifiedKFold
            
            # Combinar todos os dados (exceto teste)
            X_all = np.concatenate([self.X_train, self.X_val], axis=0)
            y_all = np.concatenate([
                np.argmax(self.y_train, axis=1),
                np.argmax(self.y_val, axis=1)
            ], axis=0)
            
            # Configurar K-Fold
            kfold = StratifiedKFold(
                n_splits=n_folds, 
                shuffle=True, 
                random_state=self.config['SEED']
            )
            
            fold_results = []
            fold_histories = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_all, y_all)):
                logger.info(f"Treinando fold {fold + 1}/{n_folds}")
                
                # Separar dados do fold
                X_train_fold = X_all[train_idx]
                y_train_fold = keras.utils.to_categorical(y_all[train_idx], len(self.class_names))
                X_val_fold = X_all[val_idx]
                y_val_fold = keras.utils.to_categorical(y_all[val_idx], len(self.class_names))
                
                # Criar datasets
                train_dataset_fold = tf.data.Dataset.from_tensor_slices(
                    (X_train_fold, y_train_fold)
                ).shuffle(
                    buffer_size=len(X_train_fold),
                    reshuffle_each_iteration=True
                ).batch(
                    self.config['BATCH_SIZE']
                ).prefetch(tf.data.AUTOTUNE)
                
                val_dataset_fold = tf.data.Dataset.from_tensor_slices(
                    (X_val_fold, y_val_fold)
                ).batch(
                    self.config['BATCH_SIZE']
                ).prefetch(tf.data.AUTOTUNE)
                
                # Criar novo modelo para o fold
                self.construir_modelo()
                
                # Treinar
                history_fold = self.model.fit(
                    train_dataset_fold,
                    epochs=n_epochs,
                    validation_data=val_dataset_fold,
                    verbose=self.config['VERBOSE'] if fold == 0 else 0  # Apenas primeiro fold verbose
                )
                
                # Avaliar
                results_fold = self.model.evaluate(val_dataset_fold, verbose=0)
                
                # Armazenar resultados
                fold_metrics = {}
                for i, metric_name in enumerate(self.model.metrics_names):
                    fold_metrics[metric_name] = float(results_fold[i])
                
                fold_results.append(fold_metrics)
                fold_histories.append(history_fold.history)
                
                logger.info(f"Fold {fold + 1} concluído: val_accuracy = {fold_metrics.get('val_accuracy', 0):.4f}")
            
            # Calcular estatísticas agregadas
            cv_results = self._calcular_estatisticas_cv(fold_results)
            
            # Salvar resultados
            self._salvar_resultados_cv(cv_results, fold_histories)
            
            logger.info(f"Validação cruzada concluída. Acurácia média: {cv_results['mean_val_accuracy']:.4f}")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Erro na validação cruzada: {e}")
            return {}
    
    def _calcular_estatisticas_cv(self, fold_results: List[Dict]) -> Dict:
        """Calcula estatísticas dos resultados da validação cruzada."""
        cv_stats = {}
        
        # Para cada métrica, calcular média e desvio padrão
        all_metrics = set()
        for fold_metrics in fold_results:
            all_metrics.update(fold_metrics.keys())
        
        for metric in all_metrics:
            values = [fold_metrics.get(metric, 0) for fold_metrics in fold_results]
            cv_stats[f'mean_{metric}'] = float(np.mean(values))
            cv_stats[f'std_{metric}'] = float(np.std(values))
            cv_stats[f'min_{metric}'] = float(np.min(values))
            cv_stats[f'max_{metric}'] = float(np.max(values))
            cv_stats[f'{metric}_values'] = values
        
        cv_stats['n_folds'] = len(fold_results)
        
        return cv_stats
    
    def _salvar_resultados_cv(self, cv_results: Dict, fold_histories: List):
        """Salva resultados da validação cruzada."""
        try:
            # Salvar resultados
            cv_path = self.resultados_dir / 'relatorios' / 'validacao_cruzada.json'
            with open(cv_path, 'w') as f:
                # Converter valores numpy
                cv_results_serializable = {}
                for key, value in cv_results.items():
                    if isinstance(value, (np.float32, np.float64)):
                        cv_results_serializable[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        cv_results_serializable[key] = value.tolist()
                    else:
                        cv_results_serializable[key] = value
                
                json.dump(cv_results_serializable, f, indent=2)
            
            # Salvar históricos dos folds
            histories_path = self.resultados_dir / 'relatorios' / 'historico_folds.pkl'
            with open(histories_path, 'wb') as f:
                pickle.dump(fold_histories, f)
            
            logger.info(f"Resultados da validação cruzada salvos")
            
        except Exception as e:
            logger.error(f"Erro ao salvar resultados CV: {e}")
    
    # ============================================================================
    # FUNÇÕES UTILITÁRIAS
    # ============================================================================
    
    def exportar_modelo(self, format: str = 'tflite') -> bool:
        """
        Exporta o modelo para diferentes formatos.
        
        Args:
            format: Formato de exportação ('tflite', 'onnx', 'savedmodel')
        
        Returns:
            True se exportado com sucesso, False caso contrário
        """
        logger.info(f"Exportando modelo para formato {format}...")
        
        try:
            gerenciador_modelos = modelo.GerenciadorModelos()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format.lower() == 'tflite':
                output_path = self.resultados_dir / 'tflite' / f'modelo_exportado_{timestamp}.tflite'
                success = gerenciador_modelos.converter_para_tflite(
                    self.model,
                    str(output_path),
                    quantize=True
                )
                
            elif format.lower() == 'onnx':
                output_path = self.resultados_dir / 'onnx' / f'modelo_exportado_{timestamp}.onnx'
                success = gerenciador_modelos.exportar_para_onnx(
                    self.model,
                    str(output_path)
                )
                
            elif format.lower() == 'savedmodel':
                output_path = self.resultados_dir / 'savedmodel' / f'modelo_{timestamp}'
                self.model.save(str(output_path), save_format='tf')
                success = True
                
            else:
                logger.error(f"Formato de exportação desconhecido: {format}")
                return False
            
            if success:
                logger.info(f"Modelo exportado com sucesso para: {output_path}")
                return True
            else:
                return False
            
        except Exception as e:
            logger.error(f"Erro ao exportar modelo: {e}")
            return False
    
    def resumo_treinamento(self) -> Dict:
        """
        Retorna um resumo completo do treinamento.
        
        Returns:
            Dicionário com resumo
        """
        resumo = {
            'configuracoes': self.config,
            'classes': self.class_names,
            'distribuicao_classes': self.class_distribution,
            'estatisticas_treinamento': self.training_stats,
            'dados': {
                'treino': len(self.X_train) if self.X_train is not None else 0,
                'validacao': len(self.X_val) if self.X_val is not None else 0,
                'teste': len(self.X_test) if self.X_test is not None else 0
            }
        }
        
        if self.model is not None:
            resumo['modelo'] = {
                'tipo': self.config['MODEL_TYPE'],
                'parametros': self.model.count_params(),
                'layers': len(self.model.layers),
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape
            }
        
        return resumo

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def treinar_modelo_completo(
    csv_path: str,
    audio_dir: Optional[str] = None,
    config: Optional[Dict] = None,
    output_dir: str = 'resultados_treinamento'
) -> TreinadorTosse:
    """
    Função principal para treinar um modelo completo.
    
    Args:
        csv_path: Caminho para CSV com rótulos
        audio_dir: Diretório com áudios
        config: Configurações de treinamento
        output_dir: Diretório para resultados
    
    Returns:
        Instância do treinador
    """
    logger.info("=" * 60)
    logger.info("INICIANDO TREINAMENTO COMPLETO")
    logger.info("=" * 60)
    
    # Inicializar treinador
    treinador = TreinadorTosse(config=config, resultados_dir=output_dir)
    
    try:
        # 1. Carregar dados
        if not treinador.carregar_dados(csv_path, audio_dir):
            logger.error("Falha ao carregar dados. Abortando.")
            return treinador
        
        # 2. Preparar dados
        treinador.preparar_dados()
        
        # 3. Construir modelo
        treinador.construir_modelo()
        
        # 4. Treinar
        history = treinador.treinar()
        
        if history is not None:
            # 5. Avaliar no conjunto de teste
            if treinador.config['EVALUATE_ON_TEST']:
                metrics = treinador.avaliar('test')
                logger.info(f"Acurácia no teste: {metrics.get('accuracy', 0):.4f}")
            
            # 6. Exportar modelo
            treinador.exportar_modelo('tflite')
            
            # 7. Gerar resumo
            resumo = treinador.resumo_treinamento()
            
            # Salvar resumo
            resumo_path = Path(output_dir) / 'relatorios' / 'resumo_treinamento.json'
            with open(resumo_path, 'w') as f:
                json.dump(resumo, f, indent=2, default=str)
            
            logger.info(f"Resumo salvo em: {resumo_path}")
            
        logger.info("=" * 60)
        logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Erro durante treinamento completo: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return treinador

# ============================================================================
# EXECUÇÃO COMO SCRIPT
# ============================================================================

if __name__ == '__main__':
    """
    Execução como script independente.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Treinar modelo de classificação de tosse')
    
    parser.add_argument('--csv', type=str, required=True,
                       help='Caminho para arquivo CSV com rótulos')
    parser.add_argument('--audio_dir', type=str, default=None,
                       help='Diretório com arquivos de áudio')
    parser.add_argument('--output_dir', type=str, default='resultados_treinamento',
                       help='Diretório para salvar resultados')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas de treinamento')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Tamanho do batch')
    parser.add_argument('--model_type', type=str, default='cnn_avancado',
                       choices=['cnn_basico', 'cnn_avancado', 'mobilenet', 
                                'efficientnet', 'cnn_lstm', 'mobile'],
                       help='Tipo de modelo a treinar')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Taxa de aprendizado')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limitar número de amostras (para teste)')
    
    args = parser.parse_args()
    
    # Configurar logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'treinamento_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Configurações personalizadas
    config_custom = {
        'EPOCHS': args.epochs,
        'BATCH_SIZE': args.batch_size,
        'MODEL_TYPE': args.model_type,
        'LEARNING_RATE': args.learning_rate
    }
    
    # Executar treinamento
    treinador = treinar_modelo_completo(
        csv_path=args.csv,
        audio_dir=args.audio_dir,
        config=config_custom,
        output_dir=args.output_dir
    )
    
    # Mostrar resumo final
    if treinador.training_stats:
        print("\n" + "=" * 60)
        print("RESUMO FINAL DO TREINAMENTO")
        print("=" * 60)
        
        resumo = treinador.resumo_treinamento()
        print(f"Classes: {resumo['classes']}")
        print(f"Amostras: Treino={resumo['dados']['treino']}, "
              f"Val={resumo['dados']['validacao']}, "
              f"Teste={resumo['dados']['teste']}")
        
        if 'modelo' in resumo:
            print(f"Modelo: {resumo['modelo']['tipo']}")
            print(f"Parâmetros: {resumo['modelo']['parametros']:,}")
        
        if 'estatisticas_treinamento' in resumo:
            stats = resumo['estatisticas_treinamento']
            print(f"Épocas: {stats.get('final_epoch', 'N/A')}")
            print(f"Acurácia final (val): {stats.get('final_val_accuracy', 0):.4f}")
            print(f"Tempo de treinamento: {stats.get('training_time', 'N/A')}")
        
        print("=" * 60)
