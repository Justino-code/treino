"""
utilitarios.py - Módulo com funções utilitárias para o projeto de classificação de tosse

Este módulo contém funções auxiliares para:
- Gerenciamento de diretórios e arquivos
- Configuração de logging
- Serialização/deserialização de dados
- Visualização de resultados
- Cálculo de métricas
"""

import os
import sys
import json
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo padrão para visualizações
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# ============================================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================================

def configurar_logging(
    nome_log: str = 'treinamento_tosse',
    nivel: str = 'INFO',
    salvar_arquivo: bool = True,
    diretorio_logs: str = 'logs'
) -> logging.Logger:
    """
    Configura o sistema de logging para o projeto.
    
    Args:
        nome_log: Nome do logger
        nivel: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        salvar_arquivo: Se True, salva logs em arquivo
        diretorio_logs: Diretório para salvar arquivos de log
    
    Returns:
        Logger configurado
    """
    # Criar diretório de logs se não existir
    os.makedirs(diretorio_logs, exist_ok=True)
    
    # Mapear nível string para constante
    nivel_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    nivel_log = nivel_map.get(nivel.upper(), logging.INFO)
    
    # Criar logger
    logger = logging.getLogger(nome_log)
    logger.setLevel(nivel_log)
    
    # Remover handlers existentes para evitar duplicação
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Formato do log
    formato = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(nivel_log)
    console_handler.setFormatter(formato)
    logger.addHandler(console_handler)
    
    # Handler para arquivo se solicitado
    if salvar_arquivo:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        caminho_log = os.path.join(diretorio_logs, f'{nome_log}_{timestamp}.log')
        
        file_handler = logging.FileHandler(caminho_log, encoding='utf-8')
        file_handler.setLevel(nivel_log)
        file_handler.setFormatter(formato)
        logger.addHandler(file_handler)
    
    logger.info(f'Logging configurado com nível: {nivel}')
    logger.info(f'Arquivo de log: {caminho_log if salvar_arquivo else "Apenas console"}')
    
    return logger

# ============================================================================
# GERENCIAMENTO DE DIRETÓRIOS E ARQUIVOS
# ============================================================================

def criar_estrutura_projeto(diretorio_base: str = '.') -> Dict[str, str]:
    """
    Cria toda a estrutura de diretórios necessária para o projeto.
    
    Args:
        diretorio_base: Diretório base para criar a estrutura
    
    Returns:
        Dicionário com caminhos dos diretórios criados
    """
    logger = logging.getLogger(__name__)
    
    estrutura = {
        'dados_brutos': 'dados/brutos',
        'dados_processados': 'dados/processados',
        'espectrogramas': 'dados/processados/espectrogramas',
        'mfcc': 'dados/processados/mfcc',
        'audios_processados': 'dados/processados/audios_processados',
        'modelos_keras': 'modelos_salvos/keras',
        'modelos_tflite': 'modelos_salvos/tflite',
        'logs_tensorboard': 'logs/tensorboard',
        'logs_metricas': 'logs/metricas',
        'notebooks': 'notebooks',
        'graficos': 'resultados/graficos',
        'relatorios': 'resultados/relatorios',
        'classificacao': 'resultados/classificacao',
        'checkpoints': 'checkpoints',
        'temp': 'temp'
    }
    
    caminhos = {}
    
    for nome, caminho_rel in estrutura.items():
        caminho_abs = os.path.join(diretorio_base, caminho_rel)
        os.makedirs(caminho_abs, exist_ok=True)
        caminhos[nome] = caminho_abs
        logger.debug(f'Diretório criado/verificado: {caminho_abs}')
    
    # Criar arquivos de configuração iniciais
    arquivos_config = {
        'dados/rotulos.csv': 'caminho_audio,rotulo,dataset,observacoes\n',
        'configuracao_projeto.json': json.dumps({
            'projeto': 'Classificador de Sons de Tosse',
            'versao': '1.0.0',
            'data_criacao': datetime.now().isoformat(),
            'classes': ['normal', 'bronquite', 'pneumonia'],
            'configuracoes_padrao': {
                'taxa_amostragem': 16000,
                'duracao_audio': 3.0,
                'n_mels': 128,
                'n_mfcc': 13,
                'shape_espectrograma': (128, 128, 3)
            }
        }, indent=2, ensure_ascii=False)
    }
    
    for caminho_arquivo, conteudo in arquivos_config.items():
        caminho_completo = os.path.join(diretorio_base, caminho_arquivo)
        with open(caminho_completo, 'w', encoding='utf-8') as f:
            f.write(conteudo)
        logger.debug(f'Arquivo criado: {caminho_completo}')
    
    logger.info(f'Estrutura do projeto criada em: {os.path.abspath(diretorio_base)}')
    
    # Salvar estrutura em arquivo
    caminho_estrutura = os.path.join(diretorio_base, 'estrutura_projeto.json')
    with open(caminho_estrutura, 'w', encoding='utf-8') as f:
        json.dump(caminhos, f, indent=2, ensure_ascii=False)
    
    return caminhos

def verificar_dependencias() -> Dict[str, Tuple[bool, str]]:
    """
    Verifica se todas as dependências necessárias estão instaladas.
    
    Returns:
        Dicionário com status de cada dependência
    """
    dependencias = {
        'tensorflow': '2.10.0',
        'librosa': '0.10.0',
        'numpy': '1.23.0',
        'pandas': '1.5.0',
        'scikit-learn': '1.2.0',
        'matplotlib': '3.6.0',
        'seaborn': '0.12.0',
        'soundfile': '0.12.0',
        'scipy': '1.10.0'
    }
    
    resultados = {}
    
    for lib, versao_min in dependencias.items():
        try:
            mod = __import__(lib)
            versao_atual = getattr(mod, '__version__', 'Desconhecida')
            
            if versao_atual != 'Desconhecida':
                # Converter versões para tuple para comparação
                def versao_para_tuple(v):
                    return tuple(map(int, v.split('.')[:3]))
                
                atual_tuple = versao_para_tuple(versao_atual)
                min_tuple = versao_para_tuple(versao_min)
                
                compativel = atual_tuple >= min_tuple
                status = 'OK' if compativel else f'Versão {versao_atual} < {versao_min}'
            else:
                compatível = True
                status = 'OK (versão desconhecida)'
            
            resultados[lib] = (compatível, f'{versao_atual} - {status}')
            
        except ImportError:
            resultados[lib] = (False, 'Não instalada')
    
    return resultados

# ============================================================================
# SERIALIZAÇÃO E ARMAZENAMENTO
# ============================================================================

def salvar_objeto(objeto: Any, caminho: str, formato: str = 'pickle') -> bool:
    """
    Salva um objeto em disco nos formatos suportados.
    
    Args:
        objeto: Objeto a ser salvo
        caminho: Caminho do arquivo
        formato: Formato de serialização ('pickle', 'numpy', 'json')
    
    Returns:
        True se salvo com sucesso, False caso contrário
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        
        if formato == 'pickle':
            with open(caminho, 'wb') as f:
                pickle.dump(objeto, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif formato == 'numpy':
            np.save(caminho, objeto)
        
        elif formato == 'json':
            # Converter numpy arrays para listas se necessário
            def converter_para_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                else:
                    return obj
            
            with open(caminho, 'w', encoding='utf-8') as f:
                json.dump(objeto, f, default=converter_para_json, indent=2)
        
        else:
            raise ValueError(f"Formato não suportado: {formato}")
        
        logger.debug(f'Objeto salvo em {caminho} (formato: {formato})')
        return True
    
    except Exception as e:
        logger.error(f'Erro ao salvar objeto em {caminho}: {e}')
        return False

def carregar_objeto(caminho: str, formato: str = None) -> Optional[Any]:
    """
    Carrega um objeto salvo em disco.
    
    Args:
        caminho: Caminho do arquivo
        formato: Formato do arquivo (inferido da extensão se None)
    
    Returns:
        Objeto carregado ou None em caso de erro
    """
    logger = logging.getLogger(__name__)
    
    try:
        if formato is None:
            # Inferir formato da extensão
            ext = os.path.splitext(caminho)[1].lower()
            if ext == '.pkl' or ext == '.pickle':
                formato = 'pickle'
            elif ext == '.npy':
                formato = 'numpy'
            elif ext == '.json':
                formato = 'json'
            else:
                raise ValueError(f"Não foi possível inferir formato da extensão {ext}")
        
        if not os.path.exists(caminho):
            logger.error(f'Arquivo não encontrado: {caminho}')
            return None
        
        if formato == 'pickle':
            with open(caminho, 'rb') as f:
                return pickle.load(f)
        
        elif formato == 'numpy':
            return np.load(caminho, allow_pickle=True)
        
        elif formato == 'json':
            with open(caminho, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        else:
            raise ValueError(f"Formato não suportado: {formato}")
    
    except Exception as e:
        logger.error(f'Erro ao carregar objeto de {caminho}: {e}')
        return None

# ============================================================================
# VISUALIZAÇÃO E ANÁLISE
# ============================================================================

def plotar_historico_treinamento(
    historico: Dict,
    salvar: bool = True,
    caminho_salvar: str = 'resultados/graficos/historico_treinamento.png',
    mostrar: bool = True
) -> plt.Figure:
    """
    Plota gráficos de perda e acurácia durante o treinamento.
    
    Args:
        historico: Dicionário com histórico de treinamento
        salvar: Se True, salva a figura
        caminho_salvar: Caminho para salvar a figura
        mostrar: Se True, mostra a figura
    
    Returns:
        Figura matplotlib
    """
    logger = logging.getLogger(__name__)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plotar perda
    if 'loss' in historico:
        axes[0].plot(historico['loss'], label='Treino', linewidth=2)
    if 'val_loss' in historico:
        axes[0].plot(historico['val_loss'], label='Validação', linewidth=2)
    
    axes[0].set_title('Perda durante o Treinamento', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Perda', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')  # Escala log para melhor visualização
    
    # Plotar acurácia
    if 'accuracy' in historico:
        axes[1].plot(historico['accuracy'], label='Treino', linewidth=2)
    if 'val_accuracy' in historico:
        axes[1].plot(historico['val_accuracy'], label='Validação', linewidth=2)
    
    axes[1].set_title('Acurácia durante o Treinamento', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Época', fontsize=12)
    axes[1].set_ylabel('Acurácia', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])  # Limite para acurácia
    
    plt.tight_layout()
    
    if salvar:
        os.makedirs(os.path.dirname(caminho_salvar), exist_ok=True)
        plt.savefig(caminho_salvar, dpi=150, bbox_inches='tight')
        logger.info(f'Gráfico salvo em: {caminho_salvar}')
    
    if mostrar:
        plt.show()
    else:
        plt.close()
    
    return fig

def plotar_matriz_confusao(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    salvar: bool = True,
    caminho_salvar: str = 'resultados/graficos/matriz_confusao.png',
    mostrar: bool = True
) -> plt.Figure:
    """
    Plota matriz de confusão.
    
    Args:
        y_true: Rótulos verdadeiros
        y_pred: Rótulos preditos
        classes: Nomes das classes
        salvar: Se True, salva a figura
        caminho_salvar: Caminho para salvar a figura
        mostrar: Se True, mostra a figura
    
    Returns:
        Figura matplotlib
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    logger = logging.getLogger(__name__)
    
    # Calcular matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalizar por linha (verdadeiros)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plotar matriz de confusão absoluta
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                ax=axes[0], cbar=False)
    axes[0].set_title('Matriz de Confusão (Absoluta)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predito', fontsize=12)
    axes[0].set_ylabel('Verdadeiro', fontsize=12)
    
    # Plotar matriz de confusão normalizada
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=classes, yticklabels=classes,
                ax=axes[1], cbar=False)
    axes[1].set_title('Matriz de Confusão (Normalizada)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predito', fontsize=12)
    axes[1].set_ylabel('Verdadeiro', fontsize=12)
    
    plt.tight_layout()
    
    if salvar:
        os.makedirs(os.path.dirname(caminho_salvar), exist_ok=True)
        plt.savefig(caminho_salvar, dpi=150, bbox_inches='tight')
        logger.info(f'Matriz de confusão salva em: {caminho_salvar}')
    
    if mostrar:
        plt.show()
    else:
        plt.close()
    
    return fig

def plotar_curva_roc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    classes: List[str],
    salvar: bool = True,
    caminho_salvar: str = 'resultados/graficos/curva_roc.png',
    mostrar: bool = True
) -> Optional[plt.Figure]:
    """
    Plota curva ROC multiclasse.
    
    Args:
        y_true: Rótulos verdadeiros (one-hot encoded)
        y_scores: Pontuações preditas (probabilidades)
        classes: Nomes das classes
        salvar: Se True, salva a figura
        caminho_salvar: Caminho para salvar a figura
        mostrar: Se True, mostra a figura
    
    Returns:
        Figura matplotlib ou None em caso de erro
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    logger = logging.getLogger(__name__)
    
    try:
        n_classes = len(classes)
        
        # Converter y_true para one-hot se necessário
        if len(y_true.shape) == 1 or y_true.shape[1] != n_classes:
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
        else:
            y_true_bin = y_true
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calcular ROC para cada classe
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plotar curva ROC para cada classe
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                   label=f'{classes[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Plotar linha de referência (aleatório)
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Aleatório (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taxa de Falsos Positivos', fontsize=12)
        ax.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
        ax.set_title('Curva ROC - Multiclasse', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if salvar:
            os.makedirs(os.path.dirname(caminho_salvar), exist_ok=True)
            plt.savefig(caminho_salvar, dpi=150, bbox_inches='tight')
            logger.info(f'Curva ROC salva em: {caminho_salvar}')
        
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    except Exception as e:
        logger.error(f'Erro ao plotar curva ROC: {e}')
        return None

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def calcular_metricas_classificacao(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str]
) -> Dict[str, Any]:
    """
    Calcula métricas de classificação multiclasse.
    
    Args:
        y_true: Rótulos verdadeiros
        y_pred: Rótulos preditos
        classes: Nomes das classes
    
    Returns:
        Dicionário com métricas calculadas
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, classification_report, confusion_matrix
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        metrics = {}
        
        # Métricas globais
        metrics['acuracia'] = accuracy_score(y_true, y_pred)
        metrics['precisao_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Métricas por classe
        metrics['precisao_por_classe'] = precision_score(y_true, y_pred, average=None, zero_division=0)
        metrics['recall_por_classe'] = recall_score(y_true, y_pred, average=None, zero_division=0)
        metrics['f1_por_classe'] = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Relatório de classificação
        metrics['relatorio'] = classification_report(
            y_true, y_pred, 
            target_names=classes,
            output_dict=True,
            zero_division=0
        )
        
        # Matriz de confusão
        metrics['matriz_confusao'] = confusion_matrix(y_true, y_pred).tolist()
        
        logger.info(f'Métricas calculadas: Acurácia = {metrics["acuracia"]:.3f}')
        
        return metrics
    
    except Exception as e:
        logger.error(f'Erro ao calcular métricas: {e}')
        return {}

def gerar_relatorio_treinamento(
    metricas: Dict[str, Any],
    parametros: Dict[str, Any],
    caminho_salvar: str = 'resultados/relatorios/relatorio_treinamento.json'
) -> bool:
    """
    Gera e salva um relatório completo do treinamento.
    
    Args:
        metricas: Dicionário com métricas do treinamento
        parametros: Dicionário com parâmetros usados
        caminho_salvar: Caminho para salvar o relatório
    
    Returns:
        True se salvo com sucesso, False caso contrário
    """
    logger = logging.getLogger(__name__)
    
    try:
        relatorio = {
            'cabecalho': {
                'projeto': 'Classificador de Sons de Tosse',
                'data_geracao': datetime.now().isoformat(),
                'versao': '1.0.0'
            },
            'parametros_treinamento': parametros,
            'metricas': metricas,
            'resumo': {
                'acuracia_geral': metricas.get('acuracia', 0),
                'melhor_classe': None,
                'pior_classe': None
            }
        }
        
        # Adicionar análise das classes
        if 'relatorio' in metricas:
            f1_por_classe = []
            for classe, dados in metricas['relatorio'].items():
                if isinstance(dados, dict) and 'f1-score' in dados:
                    f1_por_classe.append((classe, dados['f1-score']))
            
            if f1_por_classe:
                # Remover 'accuracy', 'macro avg', 'weighted avg'
                f1_por_classe = [x for x in f1_por_classe if x[0] not in ['accuracy', 'macro avg', 'weighted avg']]
                
                if f1_por_classe:
                    melhor = max(f1_por_classe, key=lambda x: x[1])
                    pior = min(f1_por_classe, key=lambda x: x[1])
                    
                    relatorio['resumo']['melhor_classe'] = {
                        'nome': melhor[0],
                        'f1_score': melhor[1]
                    }
                    relatorio['resumo']['pior_classe'] = {
                        'nome': pior[0],
                        'f1_score': pior[1]
                    }
        
        # Salvar relatório
        os.makedirs(os.path.dirname(caminho_salvar), exist_ok=True)
        with open(caminho_salvar, 'w', encoding='utf-8') as f:
            json.dump(relatorio, f, indent=2, ensure_ascii=False)
        
        logger.info(f'Relatório salvo em: {caminho_salvar}')
        return True
    
    except Exception as e:
        logger.error(f'Erro ao gerar relatório: {e}')
        return False

# ============================================================================
# FUNÇÕES DE TEMPO E PROGRESSO
# ============================================================================

class Cronometro:
    """Classe para medir tempo de execução de operações."""
    
    def __init__(self, nome: str = "Operação"):
        self.nome = nome
        self.inicio = None
        self.fim = None
    
    def __enter__(self):
        self.inicio = datetime.now()
        print(f"[{self.inicio.strftime('%H:%M:%S')}] Iniciando: {self.nome}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fim = datetime.now()
        duracao = self.fim - self.inicio
        print(f"[{self.fim.strftime('%H:%M:%S')}] Concluído: {self.nome}")
        print(f"  Duração: {duracao.total_seconds():.2f} segundos")
    
    def tempo_decorrido(self) -> float:
        """Retorna o tempo decorrido em segundos."""
        if self.inicio is None:
            return 0
        fim = self.fim if self.fim else datetime.now()
        return (fim - self.inicio).total_seconds()

def mostrar_progresso(
    iteracao: int,
    total: int,
    prefixo: str = '',
    sufixo: str = '',
    tamanho_barra: int = 50
) -> None:
    """
    Exibe uma barra de progresso no console.
    
    Args:
        iteracao: Iteração atual
        total: Total de iterações
        prefixo: Texto antes da barra
        sufixo: Texto depois da barra
        tamanho_barra: Tamanho da barra em caracteres
    """
    percentual = (iteracao + 1) / total
    barras_preenchidas = int(tamanho_barra * percentual)
    barra = '█' * barras_preenchidas + '-' * (tamanho_barra - barras_preenchidas)
    
    sys.stdout.write(f'\r{prefixo} |{barra}| {percentual:.1%} {sufixo}')
    sys.stdout.flush()
    
    if iteracao == total - 1:
        sys.stdout.write('\n')

# ============================================================================
# FUNÇÃO PRINCIPAL DE TESTE
# ============================================================================

if __name__ == '__main__':
    """
    Teste das funcionalidades do módulo utilitários.
    """
    print("=" * 60)
    print("TESTE DO MÓDULO UTILITÁRIOS")
    print("=" * 60)
    
    # 1. Configurar logging
    logger = configurar_logging('teste_utilitarios', nivel='INFO')
    logger.info("Logging configurado com sucesso!")
    
    # 2. Criar estrutura do projeto
    print("\n1. Criando estrutura do projeto...")
    caminhos = criar_estrutura_projeto()
    logger.info(f"Estrutura criada em: {caminhos.get('dados_brutos', '')}")
    
    # 3. Verificar dependências
    print("\n2. Verificando dependências...")
    deps = verificar_dependencias()
    for lib, (status, info) in deps.items():
        simbolo = "✓" if status else "✗"
        print(f"  {simbolo} {lib:15} - {info}")
    
    # 4. Testar serialização
    print("\n3. Testando serialização...")
    dados_teste = {'array': np.random.randn(10, 10), 'valor': 42, 'texto': 'teste'}
    
    with Cronometro("Serialização de dados") as timer:
        salvar_objeto(dados_teste, 'temp/teste.pkl', 'pickle')
        salvar_objeto(dados_teste, 'temp/teste.json', 'json')
    
    dados_carregados = carregar_objeto('temp/teste.pkl')
    if dados_carregados is not None:
        print(f"  Dados carregados com sucesso: {len(dados_carregados)} itens")
    
    # 5. Testar funções de visualização (dados de exemplo)
    print("\n4. Testando funções de visualização...")
    
    # Dados de exemplo para plotagem
    historico_exemplo = {
        'loss': [1.0, 0.6, 0.4, 0.3, 0.25, 0.22, 0.20, 0.18, 0.17, 0.16],
        'val_loss': [1.1, 0.7, 0.5, 0.35, 0.3, 0.28, 0.27, 0.26, 0.25, 0.24],
        'accuracy': [0.4, 0.6, 0.7, 0.75, 0.78, 0.8, 0.82, 0.83, 0.84, 0.85],
        'val_accuracy': [0.35, 0.55, 0.65, 0.7, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78]
    }
    
    y_true_exemplo = np.random.randint(0, 3, 100)
    y_pred_exemplo = np.random.randint(0, 3, 100)
    classes_exemplo = ['Normal', 'Bronquite', 'Pneumonia']
    
    plotar_historico_treinamento(historico_exemplo, salvar=False, mostrar=False)
    plotar_matriz_confusao(y_true_exemplo, y_pred_exemplo, classes_exemplo, salvar=False, mostrar=False)
    
    print("  Funções de visualização testadas com sucesso!")
    
    # 6. Testar cálculo de métricas
    print("\n5. Testando cálculo de métricas...")
    metricas = calcular_metricas_classificacao(y_true_exemplo, y_pred_exemplo, classes_exemplo)
    print(f"  Acurácia calculada: {metricas.get('acuracia', 0):.3f}")
    
    # 7. Testar barra de progresso
    print("\n6. Testando barra de progresso...")
    for i in range(100):
        mostrar_progresso(i, 100, prefixo="Progresso", sufixo=f"Item {i+1}/100")
        import time
        time.sleep(0.01)
    
    print("\n" + "=" * 60)
    print("TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    
    logger.info("Teste do módulo utilitários concluído")
