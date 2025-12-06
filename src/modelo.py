"""
modelo.py - Módulo para construção e gerenciamento de modelos de classificação de tosse

Este módulo contém funções para:
- Construção de arquiteturas de modelos (CNN, LSTM, híbridos)
- Definição de callbacks e otimizadores
- Funções de perda e métricas personalizadas
- Gerenciamento de checkpoints
- Conversão de modelos para formato móvel (TensorFlow Lite)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger, LearningRateScheduler
)
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.metrics import (
    Precision, Recall, AUC, CategoricalAccuracy
)
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES E CONFIGURAÇÕES
# ============================================================================

class ConfigModelo:
    """Configurações padrão para construção de modelos."""
    
    # Configurações de modelo
    INPUT_SHAPE = (128, 128, 3)  # Para espectrogramas como imagens
    NUM_CLASSES = 3               # normal, bronquite, pneumonia
    LEARNING_RATE = 1e-4
    DROPOUT_RATE = 0.5
    REGULARIZATION = 1e-4
    
    # Configurações de treinamento
    BATCH_SIZE = 32
    EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    
    # Configurações de callbacks
    PATIENCE_EARLY_STOP = 15
    PATIENCE_REDUCE_LR = 8
    FACTOR_REDUCE_LR = 0.5
    MIN_LEARNING_RATE = 1e-7
    
    # Configurações de arquitetura
    CNN_FILTERS = [32, 64, 128, 256]
    LSTM_UNITS = [128, 64]
    DENSE_UNITS = [256, 128, 64]

# ============================================================================
# FUNÇÕES PARA CONSTRUÇÃO DE MODELOS
# ============================================================================

class ConstrutorModelos:
    """Classe para construção de diferentes arquiteturas de modelos."""
    
    def __init__(
        self,
        input_shape: Tuple = ConfigModelo.INPUT_SHAPE,
        num_classes: int = ConfigModelo.NUM_CLASSES,
        dropout_rate: float = ConfigModelo.DROPOUT_RATE,
        regularization: float = ConfigModelo.REGULARIZATION
    ):
        """
        Inicializa o construtor de modelos.
        
        Args:
            input_shape: Shape dos dados de entrada
            num_classes: Número de classes de saída
            dropout_rate: Taxa de dropout
            regularization: Fator de regularização L2
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        
        logger.info(f"Construtor de modelos inicializado: "
                   f"input_shape={input_shape}, classes={num_classes}")
    
    def criar_modelo_cnn_basico(self) -> Model:
        """
        Cria uma CNN básica para classificação de espectrogramas.
        
        Returns:
            Modelo Keras compilado
        """
        logger.info("Criando modelo CNN básico...")
        
        model = Sequential([
            # Camada de entrada
            layers.Input(shape=self.input_shape),
            
            # Primeiro bloco convolucional
            layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=l2(self.regularization)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate * 0.5),
            
            # Segundo bloco convolucional
            layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=l2(self.regularization)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate),
            
            # Terceiro bloco convolucional
            layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=l2(self.regularization)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate),
            
            # Camadas totalmente conectadas
            layers.Flatten(),
            layers.Dense(128, activation='relu',
                        kernel_regularizer=l2(self.regularization)),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            layers.Dense(64, activation='relu',
                        kernel_regularizer=l2(self.regularization)),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate * 0.5),
            
            # Camada de saída
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def criar_modelo_cnn_avancado(self, use_residual: bool = True) -> Model:
        """
        Cria uma CNN avançada com blocos residuais opcionais.
        
        Args:
            use_residual: Se True, usa blocos residuais
        
        Returns:
            Modelo Keras compilado
        """
        logger.info("Criando modelo CNN avançado...")
        
        inputs = layers.Input(shape=self.input_shape)
        
        # Bloco inicial
        x = layers.Conv2D(32, (7, 7), padding='same', strides=2,
                         kernel_regularizer=l2(self.regularization))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        
        # Blocos intermediários
        filter_sizes = [64, 128, 256]
        
        for i, filters in enumerate(filter_sizes):
            # Bloco convolucional
            shortcut = x
            
            x = layers.Conv2D(filters, (3, 3), padding='same',
                             kernel_regularizer=l2(self.regularization))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            x = layers.Conv2D(filters, (3, 3), padding='same',
                             kernel_regularizer=l2(self.regularization))(x)
            x = layers.BatchNormalization()(x)
            
            # Conexão residual
            if use_residual and shortcut.shape[-1] == filters:
                x = layers.add([x, shortcut])
            
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(self.dropout_rate * (i + 1) / len(filter_sizes))(x)
        
        # Camadas finais
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=l2(self.regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=l2(self.regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate * 0.5)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def criar_modelo_mobilenet(self, trainable_base: bool = False) -> Model:
        """
        Cria modelo baseado em MobileNetV2 (transfer learning).
        
        Args:
            trainable_base: Se True, permite treinar a base
        
        Returns:
            Modelo Keras compilado
        """
        logger.info("Criando modelo MobileNetV2...")
        
        # Carregar MobileNetV2 pré-treinado
        base_model = keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Congelar pesos da base se especificado
        base_model.trainable = trainable_base
        
        if not trainable_base:
            for layer in base_model.layers:
                layer.trainable = False
        
        # Construir modelo completo
        inputs = layers.Input(shape=self.input_shape)
        
        # Aplicar pré-processamento específico do MobileNetV2
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # Passar pela base
        x = base_model(x, training=False)
        
        # Adicionar camadas personalizadas
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=l2(self.regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=l2(self.regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate * 0.5)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def criar_modelo_lstm(self, sequence_length: int = 100) -> Model:
        """
        Cria modelo LSTM para sequências temporais (MFCCs).
        
        Args:
            sequence_length: Comprimento das sequências de entrada
        
        Returns:
            Modelo Keras compilado
        """
        logger.info("Criando modelo LSTM...")
        
        # Shape de entrada para sequências (timesteps, features)
        input_shape = (sequence_length, 13)  # MFCCs padrão
        
        model = Sequential([
            layers.Input(shape=input_shape),
            
            # Primeira camada LSTM
            layers.LSTM(128, return_sequences=True,
                       kernel_regularizer=l2(self.regularization)),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            # Segunda camada LSTM
            layers.LSTM(64, return_sequences=False,
                       kernel_regularizer=l2(self.regularization)),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            # Camadas densas
            layers.Dense(64, activation='relu',
                        kernel_regularizer=l2(self.regularization)),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate * 0.5),
            
            # Camada de saída
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def criar_modelo_cnn_lstm_hibrido(self) -> Model:
        """
        Cria modelo híbrido CNN-LSTM para extrair features espaciais e temporais.
        
        Returns:
            Modelo Keras compilado
        """
        logger.info("Criando modelo CNN-LSTM híbrido...")
        
        inputs = layers.Input(shape=self.input_shape)
        
        # Parte CNN para extrair features espaciais
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(self.dropout_rate * 0.5)(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Redimensionar para sequência temporal
        # (batch, height, width, channels) -> (batch, timesteps, features)
        x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
        
        # Parte LSTM para modelar dependências temporais
        x = layers.LSTM(128, return_sequences=True,
                       kernel_regularizer=l2(self.regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.LSTM(64, return_sequences=False,
                       kernel_regularizer=l2(self.regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Camadas densas finais
        x = layers.Dense(64, activation='relu',
                        kernel_regularizer=l2(self.regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate * 0.5)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def criar_modelo_eficientnet(self, variant: str = 'B0') -> Model:
        """
        Cria modelo baseado em EfficientNet (transfer learning).
        
        Args:
            variant: Variante do EfficientNet (B0 a B7)
        
        Returns:
            Modelo Keras compilado
        """
        logger.info(f"Criando modelo EfficientNet{variant}...")
        
        # Mapear variante para função de construção
        efficientnet_variants = {
            'B0': keras.applications.EfficientNetB0,
            'B1': keras.applications.EfficientNetB1,
            'B2': keras.applications.EfficientNetB2,
            'B3': keras.applications.EfficientNetB3,
            'B4': keras.applications.EfficientNetB4,
            'B5': keras.applications.EfficientNetB5,
            'B6': keras.applications.EfficientNetB6,
            'B7': keras.applications.EfficientNetB7
        }
        
        if variant not in efficientnet_variants:
            logger.warning(f"Variante {variant} não encontrada. Usando B0.")
            variant = 'B0'
        
        # Carregar modelo base
        base_model = efficientnet_variants[variant](
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'
        )
        
        # Congelar pesos da base inicialmente
        base_model.trainable = False
        
        # Construir modelo completo
        inputs = layers.Input(shape=self.input_shape)
        
        # Aplicar pré-processamento específico
        x = keras.applications.efficientnet.preprocess_input(inputs)
        
        # Passar pela base
        x = base_model(x, training=False)
        
        # Adicionar camadas personalizadas
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=l2(self.regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=l2(self.regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate * 0.5)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def criar_modelo_leve_mobile(self) -> Model:
        """
        Cria modelo otimizado para dispositivos móveis.
        
        Returns:
            Modelo Keras compilado
        """
        logger.info("Criando modelo leve para mobile...")
        
        model = Sequential([
            layers.Input(shape=self.input_shape),
            
            # Primeira camada convolucional leve
            layers.SeparableConv2D(16, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate * 0.3),
            
            # Segunda camada convolucional leve
            layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate * 0.5),
            
            # Terceira camada convolucional leve
            layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate),
            
            # Camadas finais
            layers.GlobalAveragePooling2D(),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate * 0.5),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model

# ============================================================================
# FUNÇÕES DE COMPILAÇÃO E OTIMIZAÇÃO
# ============================================================================

class GerenciadorCompilacao:
    """Classe para gerenciar compilação e otimização de modelos."""
    
    @staticmethod
    def compilar_modelo(
        model: Model,
        learning_rate: float = ConfigModelo.LEARNING_RATE,
        optimizer: str = 'adam',
        loss: str = 'categorical_crossentropy',
        metrics: Optional[List[Union[str, Callable]]] = None
    ) -> Model:
        """
        Compila um modelo Keras com configurações otimizadas.
        
        Args:
            model: Modelo Keras a ser compilado
            learning_rate: Taxa de aprendizado
            optimizer: Nome do otimizador ('adam', 'sgd', 'rmsprop')
            loss: Função de perda
            metrics: Lista de métricas para monitorar
        
        Returns:
            Modelo compilado
        """
        logger.info(f"Compilando modelo com LR={learning_rate}, optimizer={optimizer}")
        
        # Configurar otimizador
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        elif optimizer.lower() == 'sgd':
            opt = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        elif optimizer.lower() == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate, rho=0.9)
        else:
            logger.warning(f"Otimizador {optimizer} não reconhecido. Usando Adam.")
            opt = Adam(learning_rate=learning_rate)
        
        # Configurar métricas padrão se não especificadas
        if metrics is None:
            metrics = [
                'accuracy',
                Precision(name='precision'),
                Recall(name='recall'),
                AUC(name='auc')
            ]
        
        # Compilar modelo
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        logger.info("Modelo compilado com sucesso")
        
        return model
    
    @staticmethod
    def criar_callbacks(
        checkpoint_dir: str = 'checkpoints',
        model_name: str = 'modelo_tosse',
        monitor: str = 'val_loss',
        mode: str = 'min',
        tensorboard_dir: str = 'logs/tensorboard',
        csv_log_path: str = 'logs/training_log.csv'
    ) -> List[tf.keras.callbacks.Callback]:
        """
        Cria uma lista de callbacks para treinamento.
        
        Args:
            checkpoint_dir: Diretório para salvar checkpoints
            model_name: Nome base para os arquivos de checkpoint
            monitor: Métrica a monitorar
            mode: 'min' para minimizar, 'max' para maximizar
            tensorboard_dir: Diretório para logs do TensorBoard
            csv_log_path: Caminho para log CSV
        
        Returns:
            Lista de callbacks
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        callbacks = []
        
        # Callback para salvar o melhor modelo
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'{model_name}_best.h5'
        )
        
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=ConfigModelo.PATIENCE_EARLY_STOP,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Redução da taxa de aprendizado
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=ConfigModelo.FACTOR_REDUCE_LR,
            patience=ConfigModelo.PATIENCE_REDUCE_LR,
            min_lr=ConfigModelo.MIN_LEARNING_RATE,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        tensorboard_callback = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        
        # CSV Logger
        csv_logger = CSVLogger(csv_log_path, append=True)
        callbacks.append(csv_logger)
        
        logger.info(f"Callbacks criados: {len(callbacks)} callbacks configurados")
        
        return callbacks
    
    @staticmethod
    def criar_learning_rate_scheduler(
        initial_lr: float = ConfigModelo.LEARNING_RATE,
        decay_steps: int = 1000,
        decay_rate: float = 0.96,
        staircase: bool = True
    ) -> Callable:
        """
        Cria um scheduler de taxa de aprendizado.
        
        Args:
            initial_lr: Taxa de aprendizado inicial
            decay_steps: Número de steps para decaimento
            decay_rate: Taxa de decaimento
            staircase: Se True, decaimento em degraus
        
        Returns:
            Função scheduler
        """
        def lr_schedule(epoch, lr):
            if staircase:
                step = epoch // decay_steps
            else:
                step = epoch / decay_steps
            
            new_lr = initial_lr * (decay_rate ** step)
            
            if epoch % 10 == 0:
                logger.info(f"Época {epoch}: LR = {new_lr:.6f}")
            
            return new_lr
        
        return LearningRateScheduler(lr_schedule, verbose=1)

# ============================================================================
# FUNÇÕES DE PERDA PERSONALIZADAS
# ============================================================================

class FuncoesPerda:
    """Classe com funções de perda personalizadas."""
    
    @staticmethod
    def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
        """
        Implementa Focal Loss para lidar com desbalanceamento de classes.
        
        Args:
            gamma: Parâmetro de foco (gamma=0 é igual a cross-entropy)
            alpha: Parâmetro de balanceamento
        
        Returns:
            Função de perda
        """
        def focal_loss_fn(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            
            # Calcular cross-entropy
            cross_entropy = -y_true * tf.math.log(y_pred)
            
            # Calcular fator de foco
            weight = alpha * tf.pow(1.0 - y_pred, gamma)
            
            # Focal loss
            loss = weight * cross_entropy
            
            return tf.reduce_sum(loss, axis=-1)
        
        return focal_loss_fn
    
    @staticmethod
    def weighted_categorical_crossentropy(class_weights: Dict[int, float]):
        """
        Cross-entropy com pesos diferentes para cada classe.
        
        Args:
            class_weights: Dicionário com pesos para cada classe
        
        Returns:
            Função de perda
        """
        def weighted_loss_fn(y_true, y_pred):
            # Converter pesos para tensor
            weights = tf.constant([class_weights.get(i, 1.0) 
                                  for i in range(len(class_weights))])
            
            # Calcular perda padrão
            loss = keras.losses.categorical_crossentropy(y_true, y_pred)
            
            # Aplicar pesos
            class_indices = tf.argmax(y_true, axis=-1)
            weight_tensor = tf.gather(weights, class_indices)
            
            return loss * weight_tensor
        
        return weighted_loss_fn
    
    @staticmethod
    def dice_loss(smooth: float = 1e-6):
        """
        Dice Loss para problemas de segmentação/classificação.
        
        Args:
            smooth: Termo de suavização para evitar divisão por zero
        
        Returns:
            Função de perda
        """
        def dice_loss_fn(y_true, y_pred):
            y_true_f = tf.reshape(y_true, [-1])
            y_pred_f = tf.reshape(y_pred, [-1])
            
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            
            dice = (2. * intersection + smooth) / \
                   (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
            
            return 1.0 - dice
        
        return dice_loss_fn
    
    @staticmethod
    def combined_loss(alpha: float = 0.5, beta: float = 0.5):
        """
        Combinação de focal loss e dice loss.
        
        Args:
            alpha: Peso para focal loss
            beta: Peso para dice loss
        
        Returns:
            Função de perda combinada
        """
        focal_fn = FuncoesPerda.focal_loss()
        dice_fn = FuncoesPerda.dice_loss()
        
        def combined_loss_fn(y_true, y_pred):
            focal = focal_fn(y_true, y_pred)
            dice = dice_fn(y_true, y_pred)
            
            return alpha * focal + beta * dice
        
        return combined_loss_fn

# ============================================================================
# FUNÇÕES DE MÉTRICAS PERSONALIZADAS
# ============================================================================

class MetricasPersonalizadas:
    """Classe com métricas personalizadas para classificação de tosse."""
    
    @staticmethod
    def f1_score():
        """
        Calcula F1-Score (média harmônica de precisão e recall).
        
        Returns:
            Função métrica
        """
        def f1_fn(y_true, y_pred):
            # Converter predições para classes
            y_pred_class = tf.argmax(y_pred, axis=-1)
            y_true_class = tf.argmax(y_true, axis=-1)
            
            # Calcular precisão e recall
            precision = tf.keras.metrics.Precision()(y_true_class, y_pred_class)
            recall = tf.keras.metrics.Recall()(y_true_class, y_pred_class)
            
            # Calcular F1
            f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
            
            return f1
        
        return f1_fn
    
    @staticmethod
    def balanced_accuracy():
        """
        Calcula acurácia balanceada (média das acurácias por classe).
        
        Returns:
            Função métrica
        """
        def balanced_acc_fn(y_true, y_pred):
            y_pred_class = tf.argmax(y_pred, axis=-1)
            y_true_class = tf.argmax(y_true, axis=-1)
            
            # Contar acertos por classe
            classes = tf.unique(y_true_class)[0]
            accuracies = []
            
            for cls in classes:
                mask = tf.equal(y_true_class, cls)
                true_positives = tf.reduce_sum(
                    tf.cast(tf.equal(y_pred_class[mask], cls), tf.float32)
                )
                total_class = tf.reduce_sum(tf.cast(mask, tf.float32))
                
                class_acc = true_positives / (total_class + 1e-7)
                accuracies.append(class_acc)
            
            return tf.reduce_mean(accuracies)
        
        return balanced_acc_fn
    
    @staticmethod
    def specificity():
        """
        Calcula especificidade (true negative rate).
        
        Returns:
            Função métrica
        """
        def specificity_fn(y_true, y_pred):
            y_pred_class = tf.argmax(y_pred, axis=-1)
            y_true_class = tf.argmax(y_true, axis=-1)
            
            # Para cada classe, calcular especificidade
            classes = tf.range(tf.shape(y_pred)[-1])
            specificities = []
            
            for cls in classes:
                # True negatives: amostras não da classe preditas como não da classe
                not_cls_true = tf.not_equal(y_true_class, cls)
                not_cls_pred = tf.not_equal(y_pred_class, cls)
                true_negatives = tf.reduce_sum(
                    tf.cast(tf.logical_and(not_cls_true, not_cls_pred), tf.float32)
                )
                
                # False positives: amostras não da classe preditas como da classe
                false_positives = tf.reduce_sum(
                    tf.cast(tf.logical_and(not_cls_true, tf.equal(y_pred_class, cls)), tf.float32)
                )
                
                spec = true_negatives / (true_negatives + false_positives + 1e-7)
                specificities.append(spec)
            
            return tf.reduce_mean(specificities)
        
        return specificity_fn

# ============================================================================
# GESTÃO DE MODELOS E CHECKPOINTS
# ============================================================================

class GerenciadorModelos:
    """Classe para gerenciar modelos e checkpoints."""
    
    @staticmethod
    def salvar_modelo(
        model: Model,
        path: str,
        save_format: str = 'h5',
        include_optimizer: bool = True
    ) -> bool:
        """
        Salva um modelo Keras em disco.
        
        Args:
            model: Modelo Keras a ser salvo
            path: Caminho para salvar
            save_format: Formato de salvamento ('h5' ou 'keras')
            include_optimizer: Se True, salva estado do otimizador
        
        Returns:
            True se salvo com sucesso, False caso contrário
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            if save_format.lower() == 'h5':
                model.save(path, save_format='h5', include_optimizer=include_optimizer)
            else:
                model.save(path, save_format='keras', include_optimizer=include_optimizer)
            
            logger.info(f"Modelo salvo em: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
            return False
    
    @staticmethod
    def carregar_modelo(
        path: str,
        custom_objects: Optional[Dict] = None
    ) -> Optional[Model]:
        """
        Carrega um modelo Keras do disco.
        
        Args:
            path: Caminho do modelo salvo
            custom_objects: Dicionário com objetos personalizados
        
        Returns:
            Modelo carregado ou None em caso de erro
        """
        try:
            if not os.path.exists(path):
                logger.error(f"Arquivo do modelo não encontrado: {path}")
                return None
            
            if custom_objects is None:
                custom_objects = {}
            
            # Adicionar funções personalizadas
            custom_objects.update({
                'focal_loss': FuncoesPerda.focal_loss(),
                'f1_score': MetricasPersonalizadas.f1_score(),
                'balanced_accuracy': MetricasPersonalizadas.balanced_accuracy(),
                'specificity': MetricasPersonalizadas.specificity()
            })
            
            model = keras.models.load_model(path, custom_objects=custom_objects)
            logger.info(f"Modelo carregado de: {path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return None
    
    @staticmethod
    def converter_para_tflite(
        model: Model,
        output_path: str,
        quantize: bool = True,
        optimization: str = 'DEFAULT'
    ) -> bool:
        """
        Converte um modelo Keras para TensorFlow Lite.
        
        Args:
            model: Modelo Keras
            output_path: Caminho de saída para o arquivo .tflite
            quantize: Se True, aplica quantização
            optimization: Nível de otimização
        
        Returns:
            True se convertido com sucesso, False caso contrário
        """
        try:
            # Criar conversor
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Configurar otimizações
            if quantize:
                converter.optimizations = [getattr(tf.lite.Optimize, optimization)]
                
                # Configurar quantização específica
                converter.target_spec.supported_types = [tf.float16]
                converter._experimental_lower_tensor_list_ops = False
            
            # Converter
            tflite_model = converter.convert()
            
            # Salvar
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Modelo convertido para TFLite: {output_path}")
            
            # Calcular tamanho do modelo
            size_kb = len(tflite_model) / 1024
            logger.info(f"Tamanho do modelo TFLite: {size_kb:.1f} KB")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao converter para TFLite: {e}")
            return False
    
    @staticmethod
    def exportar_para_onnx(
        model: Model,
        output_path: str,
        opset_version: int = 13
    ) -> bool:
        """
        Exporta um modelo Keras para formato ONNX.
        
        Args:
            model: Modelo Keras
            output_path: Caminho de saída para o arquivo .onnx
            opset_version: Versão do ONNX opset
        
        Returns:
            True se exportado com sucesso, False caso contrário
        """
        try:
            # Nota: Requer tf2onnx instalado
            import tf2onnx
            
            # Converter modelo
            model_proto, _ = tf2onnx.convert.from_keras(
                model,
                opset=opset_version,
                output_path=output_path
            )
            
            logger.info(f"Modelo exportado para ONNX: {output_path}")
            return True
            
        except ImportError:
            logger.error("tf2onnx não instalado. Instale com: pip install tf2onnx")
            return False
        except Exception as e:
            logger.error(f"Erro ao exportar para ONNX: {e}")
            return False

# ============================================================================
# FUNÇÃO PRINCIPAL DE TESTE
# ============================================================================

if __name__ == '__main__':
    """
    Teste das funcionalidades do módulo de modelos.
    """
    print("=" * 60)
    print("TESTE DO MÓDULO DE MODELOS")
    print("=" * 60)
    
    # Configurar logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Criar diretórios de teste
    os.makedirs('test_models', exist_ok=True)
    os.makedirs('test_checkpoints', exist_ok=True)
    
    # Testar construtor de modelos
    print("\n1. Testando construtor de modelos...")
    
    construtor = ConstrutorModelos()
    
    # Criar diferentes modelos
    modelos = {
        'cnn_basico': construtor.criar_modelo_cnn_basico(),
        'cnn_avancado': construtor.criar_modelo_cnn_avancado(),
        'mobile': construtor.criar_modelo_leve_mobile(),
        'cnn_lstm': construtor.criar_modelo_cnn_lstm_hibrido()
    }
    
    print(f"Modelos criados: {len(modelos)}")
    
    # Testar compilação
    print("\n2. Testando compilação de modelos...")
    
    gerenciador_compilacao = GerenciadorCompilacao()
    
    for nome, modelo in modelos.items():
        print(f"  Compilando {nome}...")
        modelo = gerenciador_compilacao.compilar_modelo(
            modelo,
            learning_rate=1e-4,
            optimizer='adam'
        )
        
        # Mostrar resumo
        print(f"  Parâmetros: {modelo.count_params():,}")
        print(f"  Layers: {len(modelo.layers)}")
    
    # Testar callbacks
    print("\n3. Testando criação de callbacks...")
    
    callbacks = gerenciador_compilacao.criar_callbacks(
        checkpoint_dir='test_checkpoints',
        model_name='test_model'
    )
    
    print(f"  Callbacks criados: {len(callbacks)}")
    for cb in callbacks:
        print(f"    - {cb.__class__.__name__}")
    
    # Testar funções de perda
    print("\n4. Testando funções de perda personalizadas...")
    
    perdas = FuncoesPerda()
    
    # Criar dados de teste
    y_true_test = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred_test = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    
    # Testar focal loss
    focal_loss_fn = perdas.focal_loss(gamma=2.0, alpha=0.25)
    loss_value = focal_loss_fn(
        tf.constant(y_true_test, dtype=tf.float32),
        tf.constant(y_pred_test, dtype=tf.float32)
    )
    print(f"  Focal Loss: {loss_value.numpy()}")
    
    # Testar métricas personalizadas
    print("\n5. Testando métricas personalizadas...")
    
    metricas = MetricasPersonalizadas()
    f1_metric = metricas.f1_score()
    
    f1_value = f1_metric(
        tf.constant(y_true_test, dtype=tf.float32),
        tf.constant(y_pred_test, dtype=tf.float32)
    )
    print(f"  F1-Score: {f1_value:.4f}")
    
    # Testar gerenciador de modelos
    print("\n6. Testando gerenciador de modelos...")
    
    gerenciador = GerenciadorModelos()
    
    # Salvar modelo
    modelo_teste = modelos['cnn_basico']
    caminho_salvo = 'test_models/modelo_teste.h5'
    
    if gerenciador.salvar_modelo(modelo_teste, caminho_salvo):
        print(f"  Modelo salvo em: {caminho_salvo}")
    
    # Carregar modelo
    modelo_carregado = gerenciador.carregar_modelo(caminho_salvo)
    if modelo_carregado:
        print(f"  Modelo carregado com sucesso")
        print(f"  Shape de entrada: {modelo_carregado.input_shape}")
        print(f"  Shape de saída: {modelo_carregado.output_shape}")
    
    # Testar conversão para TFLite
    print("\n7. Testando conversão para TFLite...")
    
    tflite_path = 'test_models/modelo_teste.tflite'
    if gerenciador.converter_para_tflite(modelo_teste, tflite_path, quantize=True):
        print(f"  Modelo TFLite salvo em: {tflite_path}")
        
        # Verificar tamanho
        if os.path.exists(tflite_path):
            size_kb = os.path.getsize(tflite_path) / 1024
            print(f"  Tamanho do arquivo: {size_kb:.1f} KB")
    
    # Limpar arquivos de teste
    print("\n8. Limpando arquivos de teste...")
    
    import shutil
    if os.path.exists('test_checkpoints'):
        shutil.rmtree('test_checkpoints')
        print("  Checkpoints removidos")
    
    if os.path.exists('test_models'):
        shutil.rmtree('test_models')
        print("  Modelos de teste removidos")
    
    print("\n" + "=" * 60)
    print("TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
