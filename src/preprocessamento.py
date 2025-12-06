"""
preprocessamento.py - Módulo de pré-processamento de áudio para classificação de tosse

Este módulo contém funções para:
- Carregamento e normalização de áudio
- Remoção de ruído e filtragem
- Segmentação e detecção de eventos
- Aumento de dados (data augmentation)
- Extração de características básicas
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
from scipy.ndimage import median_filter
from typing import Dict, List, Tuple, Optional, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES E CONFIGURAÇÕES
# ============================================================================

class ConfigAudio:
    """Configurações padrão para processamento de áudio."""
    
    # Taxa de amostragem padrão (Hz)
    SR_PADRAO = 16000
    
    # Duração padrão dos áudios (segundos)
    DURACAO_PADRAO = 3.0
    
    # Limites de frequência para filtro passa-banda (Hz)
    FREQ_MIN = 100    # Frequência mínima relevante para tosse
    FREQ_MAX = 4000   # Frequência máxima relevante para tosse
    
    # Parâmetros para detecção de atividade
    LIMIAR_ATIVIDADE = 0.05  # Limiar para detecção de atividade
    DURACAO_MINIMA = 0.5     # Duração mínima de um evento de tosse (s)
    
    # Configurações de filtros
    ORDEM_FILTRO = 4         # Ordem do filtro Butterworth
    TAMANHO_JANELA_WIENER = 256  # Tamanho da janela para filtro de Wiener
    TAMANHO_MEDIANA = 5      # Tamanho do filtro de mediana

# ============================================================================
# CLASSE PRINCIPAL DE PRÉ-PROCESSAMENTO
# ============================================================================

class PreprocessadorTosse:
    """
    Classe principal para pré-processamento de áudio de tosse.
    
    Esta classe implementa um pipeline completo de pré-processamento
    específico para sons de tosse, incluindo remoção de ruído, filtragem
    e normalização.
    """
    
    def __init__(
        self,
        sr: int = ConfigAudio.SR_PADRAO,
        duracao_alvo: float = ConfigAudio.DURACAO_PADRAO,
        aplicar_filtros: bool = True,
        remover_ruido: bool = True,
        normalizar: bool = True
    ):
        """
        Inicializa o pré-processador.
        
        Args:
            sr: Taxa de amostragem (Hz)
            duracao_alvo: Duração alvo para os áudios (segundos)
            aplicar_filtros: Se True, aplica filtros de pré-processamento
            remover_ruido: Se True, remove ruído dos áudios
            normalizar: Se True, normaliza a amplitude
        """
        self.sr = sr
        self.duracao_alvo = duracao_alvo
        self.amostras_alvo = int(sr * duracao_alvo)
        self.aplicar_filtros = aplicar_filtros
        self.remover_ruido = remover_ruido
        self.normalizar = normalizar
        
        logger.info(f"Preprocessador inicializado: SR={sr}Hz, Duração={duracao_alvo}s")
    
    # ============================================================================
    # MÉTODOS BÁSICOS DE CARREGAMENTO E SALVAMENTO
    # ============================================================================
    
    def carregar_audio(
        self,
        caminho_audio: str,
        sr: Optional[int] = None,
        mono: bool = True,
        duracao_maxima: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Carrega um arquivo de áudio com tratamento de erros robusto.
        
        Args:
            caminho_audio: Caminho para o arquivo de áudio
            sr: Taxa de amostragem desejada (None para usar original)
            mono: Se True, converte para mono
            duracao_maxima: Duração máxima a carregar (segundos)
        
        Returns:
            Tuple (audio, sr): Áudio carregado e taxa de amostragem
        
        Raises:
            FileNotFoundError: Se o arquivo não existir
            ValueError: Se o arquivo estiver corrompido ou vazio
        """
        if not os.path.exists(caminho_audio):
            logger.error(f"Arquivo não encontrado: {caminho_audio}")
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho_audio}")
        
        try:
            # Tentar carregar com librosa primeiro
            if sr is None:
                sr = self.sr
            
            # Configurar parâmetros de carregamento
            kwargs = {
                'sr': sr,
                'mono': mono,
                'duration': duracao_maxima
            }
            
            audio, sr_carregado = librosa.load(caminho_audio, **kwargs)
            
            # Verificar se o áudio não está vazio
            if len(audio) == 0:
                logger.error(f"Áudio vazio: {caminho_audio}")
                raise ValueError(f"Áudio vazio: {caminho_audio}")
            
            # Verificar se o áudio contém apenas ruído (amplitude muito baixa)
            amplitude_max = np.max(np.abs(audio))
            if amplitude_max < 1e-6:
                logger.warning(f"Amplitude muito baixa no áudio {caminho_audio}: {amplitude_max}")
            
            logger.debug(f"Áudio carregado: {caminho_audio} "
                        f"shape={audio.shape}, SR={sr_carregado}, "
                        f"duracao={len(audio)/sr_carregado:.2f}s")
            
            return audio, sr_carregado
            
        except Exception as e:
            logger.error(f"Erro ao carregar áudio {caminho_audio}: {e}")
            
            # Tentar método alternativo com soundfile
            try:
                logger.warning(f"Tentando carregar com soundfile: {caminho_audio}")
                audio, sr_carregado = sf.read(caminho_audio)
                
                # Converter para mono se necessário
                if len(audio.shape) > 1 and mono:
                    audio = np.mean(audio, axis=1)
                
                # Redimensionar taxa de amostragem se necessário
                if sr is not None and sr_carregado != sr:
                    from scipy import interpolate
                    duracao_original = len(audio) / sr_carregado
                    x_original = np.linspace(0, duracao_original, len(audio))
                    x_novo = np.linspace(0, duracao_original, int(duracao_original * sr))
                    
                    f = interpolate.interp1d(x_original, audio, kind='linear')
                    audio = f(x_novo)
                    sr_carregado = sr
                
                logger.info(f"Áudio carregado com soundfile: {caminho_audio}")
                return audio, sr_carregado
                
            except Exception as e2:
                logger.error(f"Falha ao carregar com soundfile: {e2}")
                raise ValueError(f"Não foi possível carregar o áudio {caminho_audio}: {e}")
    
    def salvar_audio(
        self,
        audio: np.ndarray,
        caminho_saida: str,
        sr: Optional[int] = None
    ) -> bool:
        """
        Salva áudio em formato WAV.
        
        Args:
            audio: Sinal de áudio
            caminho_saida: Caminho para salvar
            sr: Taxa de amostragem (usa self.sr se None)
        
        Returns:
            True se salvo com sucesso, False caso contrário
        """
        try:
            if sr is None:
                sr = self.sr
            
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
            
            # Salvar áudio
            sf.write(caminho_saida, audio, sr)
            
            logger.debug(f"Áudio salvo: {caminho_saida} "
                        f"shape={audio.shape}, SR={sr}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar áudio em {caminho_saida}: {e}")
            return False
    
    # ============================================================================
    # NORMALIZAÇÃO E PRÉ-PROCESSAMENTO BÁSICO
    # ============================================================================
    
    def normalizar_amplitude(
        self,
        audio: np.ndarray,
        metodo: str = 'peak',
        nivel_dB: float = -3.0
    ) -> np.ndarray:
        """
        Normaliza a amplitude do áudio.
        
        Args:
            audio: Sinal de áudio
            metodo: Método de normalização ('peak', 'rms', 'loudness')
            nivel_dB: Nível alvo em dB (apenas para métodos relevantes)
        
        Returns:
            Áudio normalizado
        """
        if not self.normalizar or len(audio) == 0:
            return audio
        
        try:
            if metodo == 'peak':
                # Normalização por pico (amplitude máxima = 1.0)
                max_abs = np.max(np.abs(audio))
                if max_abs > 1e-6:  # Evitar divisão por zero
                    audio_norm = audio / max_abs
                else:
                    audio_norm = audio
            
            elif metodo == 'rms':
                # Normalização por RMS
                rms = np.sqrt(np.mean(audio ** 2))
                if rms > 1e-6:
                    audio_norm = audio / rms
                else:
                    audio_norm = audio
            
            elif metodo == 'loudness':
                # Normalização por loudness (em desenvolvimento)
                # Implementação básica - normalização por percentil
                percentil = 0.95
                limiar = np.percentile(np.abs(audio), percentil * 100)
                if limiar > 1e-6:
                    audio_norm = audio / limiar
                else:
                    audio_norm = audio
            
            else:
                logger.warning(f"Método de normalização '{metodo}' não reconhecido. "
                              f"Usando normalização por pico.")
                audio_norm = self.normalizar_amplitude(audio, 'peak')
            
            logger.debug(f"Normalização '{metodo}' aplicada: "
                        f"max_antes={np.max(np.abs(audio)):.3f}, "
                        f"max_depois={np.max(np.abs(audio_norm)):.3f}")
            
            return audio_norm
            
        except Exception as e:
            logger.error(f"Erro na normalização: {e}")
            return audio
    
    def remover_offset_dc(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove componente DC (offset) do áudio.
        
        Args:
            audio: Sinal de áudio
        
        Returns:
            Áudio sem offset DC
        """
        try:
            offset = np.mean(audio)
            audio_sem_dc = audio - offset
            
            logger.debug(f"Offset DC removido: valor={offset:.6f}")
            
            return audio_sem_dc
            
        except Exception as e:
            logger.error(f"Erro ao remover offset DC: {e}")
            return audio
    
    def aplicar_preenfase(self, audio: np.ndarray, coeficiente: float = 0.97) -> np.ndarray:
        """
        Aplica pré-ênfase para realçar altas frequências.
        
        Args:
            audio: Sinal de áudio
            coeficiente: Coeficiente de pré-ênfase (tipicamente 0.95-0.97)
        
        Returns:
            Áudio com pré-ênfase
        """
        try:
            audio_pre = np.append(audio[0], audio[1:] - coeficiente * audio[:-1])
            
            logger.debug(f"Pré-ênfase aplicada: coeficiente={coeficiente}")
            
            return audio_pre
            
        except Exception as e:
            logger.error(f"Erro ao aplicar pré-ênfase: {e}")
            return audio
    
    # ============================================================================
    # FILTRAGEM E REMOÇÃO DE RUÍDO
    # ============================================================================
    
    def aplicar_filtro_passa_banda(
        self,
        audio: np.ndarray,
        freq_min: float = None,
        freq_max: float = None,
        ordem: int = None
    ) -> np.ndarray:
        """
        Aplica filtro Butterworth passa-banda.
        
        Args:
            audio: Sinal de áudio
            freq_min: Frequência de corte inferior (Hz)
            freq_max: Frequência de corte superior (Hz)
            ordem: Ordem do filtro
        
        Returns:
            Áudio filtrado
        """
        if not self.aplicar_filtros or len(audio) < 100:
            return audio
        
        try:
            if freq_min is None:
                freq_min = ConfigAudio.FREQ_MIN
            if freq_max is None:
                freq_max = ConfigAudio.FREQ_MAX
            if ordem is None:
                ordem = ConfigAudio.ORDEM_FILTRO
            
            # Calcular frequências normalizadas
            nyquist = 0.5 * self.sr
            low = freq_min / nyquist
            high = freq_max / nyquist
            
            # Verificar limites
            if low <= 0 or high >= 1 or low >= high:
                logger.warning(f"Frequências do filtro inválidas: low={low}, high={high}")
                return audio
            
            # Projetar e aplicar filtro Butterworth
            b, a = signal.butter(ordem, [low, high], btype='band')
            audio_filtrado = signal.filtfilt(b, a, audio)
            
            logger.debug(f"Filtro passa-banda aplicado: "
                        f"{freq_min}-{freq_max}Hz, ordem={ordem}")
            
            return audio_filtrado
            
        except Exception as e:
            logger.error(f"Erro ao aplicar filtro passa-banda: {e}")
            return audio
    
    def aplicar_filtro_wiener(self, audio: np.ndarray, tamanho_janela: int = None) -> np.ndarray:
        """
        Aplica filtro de Wiener para redução de ruído.
        
        Args:
            audio: Sinal de áudio
            tamanho_janela: Tamanho da janela para estimativa de ruído
        
        Returns:
            Áudio com ruído reduzido
        """
        if not self.remover_ruido or len(audio) < 100:
            return audio
        
        try:
            if tamanho_janela is None:
                tamanho_janela = ConfigAudio.TAMANHO_JANELA_WIENER
            
            # Aplicar filtro de Wiener
            audio_sem_ruido = signal.wiener(audio, mysize=tamanho_janela)
            
            logger.debug(f"Filtro de Wiener aplicado: janela={tamanho_janela}")
            
            return audio_sem_ruido
            
        except Exception as e:
            logger.error(f"Erro ao aplicar filtro de Wiener: {e}")
            return audio
    
    def aplicar_filtro_mediana(self, audio: np.ndarray, tamanho: int = None) -> np.ndarray:
        """
        Aplica filtro de mediana para remover ruídos impulsivos.
        
        Args:
            audio: Sinal de áudio
            tamanho: Tamanho do filtro de mediana
        
        Returns:
            Áudio filtrado
        """
        if not self.remover_ruido or len(audio) < 10:
            return audio
        
        try:
            if tamanho is None:
                tamanho = ConfigAudio.TAMANHO_MEDIANA
            
            # Aplicar filtro de mediana
            audio_filtrado = median_filter(audio, size=tamanho)
            
            logger.debug(f"Filtro de mediana aplicado: tamanho={tamanho}")
            
            return audio_filtrado
            
        except Exception as e:
            logger.error(f"Erro ao aplicar filtro de mediana: {e}")
            return audio
    
    def remover_ruido_espectral(
        self,
        audio: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_iter: int = 10
    ) -> np.ndarray:
        """
        Remove ruído usando subtração espectral.
        
        Args:
            audio: Sinal de áudio
            n_fft: Tamanho da FFT
            hop_length: Hop length para STFT
            n_iter: Número de iterações para estimativa de ruído
        
        Returns:
            Áudio com ruído reduzido
        """
        if not self.remover_ruido or len(audio) < n_fft:
            return audio
        
        try:
            # Estimar ruído usando primeiras n_iter frames
            D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            
            # Estimar espectro de ruído (média das primeiras frames)
            if D.shape[1] > n_iter:
                ruido_est = np.mean(np.abs(D[:, :n_iter]), axis=1, keepdims=True)
            else:
                ruido_est = np.mean(np.abs(D), axis=1, keepdims=True)
            
            # Aplicar subtração espectral
            magnitude = np.abs(D)
            fase = np.angle(D)
            
            # Subtrair ruído estimado (com limiar mínimo)
            magnitude_sub = magnitude - ruido_est
            magnitude_sub = np.maximum(magnitude_sub, 0.1 * ruido_est)
            
            # Reconstruir sinal
            D_reconstruido = magnitude_sub * np.exp(1j * fase)
            audio_reconstruido = librosa.istft(D_reconstruido, hop_length=hop_length)
            
            # Garantir mesmo comprimento
            if len(audio_reconstruido) > len(audio):
                audio_reconstruido = audio_reconstruido[:len(audio)]
            elif len(audio_reconstruido) < len(audio):
                audio_reconstruido = np.pad(audio_reconstruido, 
                                           (0, len(audio) - len(audio_reconstruido)))
            
            logger.debug(f"Remoção espectral de ruído aplicada: "
                        f"n_fft={n_fft}, hop={hop_length}")
            
            return audio_reconstruido
            
        except Exception as e:
            logger.error(f"Erro na remoção espectral de ruído: {e}")
            return audio
    
    # ============================================================================
    # SEGMENTAÇÃO E DETECÇÃO DE EVENTOS
    # ============================================================================
    
    def detectar_atividade(
        self,
        audio: np.ndarray,
        limiar: float = None,
        duracao_minima: float = None
    ) -> List[Tuple[int, int]]:
        """
        Detecta segmentos ativos no áudio (onde há tosse).
        
        Args:
            audio: Sinal de áudio
            limiar: Limiar para detecção (fração da amplitude máxima)
            duracao_minima: Duração mínima de um segmento ativo (segundos)
        
        Returns:
            Lista de tuplas (inicio, fim) em amostras
        """
        if len(audio) == 0:
            return []
        
        try:
            if limiar is None:
                limiar = ConfigAudio.LIMIAR_ATIVIDADE
            if duracao_minima is None:
                duracao_minima = ConfigAudio.DURACAO_MINIMA
            
            # Calcular envelope de energia
            energia = librosa.feature.rms(
                y=audio,
                frame_length=2048,
                hop_length=512
            )[0]
            
            # Normalizar energia
            if np.max(energia) > 0:
                energia_norm = energia / np.max(energia)
            else:
                return []
            
            # Aplicar limiar
            ativo = energia_norm > limiar
            
            # Encontrar segmentos ativos
            segmentos = []
            inicio = None
            
            for i, estado in enumerate(ativo):
                if estado and inicio is None:
                    inicio = i
                elif not estado and inicio is not None:
                    fim = i
                    
                    # Converter de frames para amostras
                    inicio_amostras = inicio * 512
                    fim_amostras = fim * 512
                    
                    # Verificar duração mínima
                    duracao_seg = (fim_amostras - inicio_amostras) / self.sr
                    if duracao_seg >= duracao_minima:
                        segmentos.append((inicio_amostras, fim_amostras))
                    
                    inicio = None
            
            # Lidar com segmento que termina no final
            if inicio is not None:
                fim_amostras = len(ativo) * 512
                duracao_seg = (fim_amostras - inicio * 512) / self.sr
                if duracao_seg >= duracao_minima:
                    segmentos.append((inicio * 512, fim_amostras))
            
            logger.debug(f"Detectados {len(segmentos)} segmentos ativos")
            
            return segmentos
            
        except Exception as e:
            logger.error(f"Erro na detecção de atividade: {e}")
            return []
    
    def extrair_segmento_principal(
        self,
        audio: np.ndarray,
        n_segmentos: int = 1
    ) -> np.ndarray:
        """
        Extrai o segmento mais energético do áudio.
        
        Args:
            audio: Sinal de áudio
            n_segmentos: Número de segmentos a extrair
        
        Returns:
            Segmento(s) extraído(s) concatenados
        """
        if len(audio) == 0:
            return audio
        
        try:
            # Detectar todos os segmentos ativos
            segmentos = self.detectar_atividade(audio)
            
            if not segmentos:
                # Se não detectar segmentos, usar áudio completo
                return self.ajustar_comprimento(audio)
            
            # Calcular energia de cada segmento
            energias = []
            for inicio, fim in segmentos:
                segmento = audio[inicio:fim]
                energia = np.sum(segmento ** 2)
                energias.append(energia)
            
            # Ordenar por energia (decrescente)
            segmentos_ordenados = sorted(zip(segmentos, energias), 
                                        key=lambda x: x[1], 
                                        reverse=True)
            
            # Extrair N segmentos mais energéticos
            segmentos_selecionados = []
            total_amostras = 0
            
            for (inicio, fim), _ in segmentos_ordenados[:n_segmentos]:
                segmento = audio[inicio:fim]
                segmentos_selecionados.append(segmento)
                total_amostras += len(segmento)
                
                # Se já temos amostras suficientes, parar
                if total_amostras >= self.amostras_alvo:
                    break
            
            # Concatenar segmentos
            if segmentos_selecionados:
                audio_concatenado = np.concatenate(segmentos_selecionados)
            else:
                audio_concatenado = audio
            
            # Ajustar comprimento
            audio_ajustado = self.ajustar_comprimento(audio_concatenado)
            
            logger.debug(f"Segmento principal extraído: "
                        f"{len(segmentos_selecionados)} segmentos, "
                        f"comprimento={len(audio_ajustado)/self.sr:.2f}s")
            
            return audio_ajustado
            
        except Exception as e:
            logger.error(f"Erro ao extrair segmento principal: {e}")
            return self.ajustar_comprimento(audio)
    
    def ajustar_comprimento(
        self,
        audio: np.ndarray,
        comprimento_alvo: Optional[int] = None
    ) -> np.ndarray:
        """
        Ajusta o comprimento do áudio para um valor fixo.
        
        Args:
            audio: Sinal de áudio
            comprimento_alvo: Comprimento alvo em amostras
        
        Returns:
            Áudio com comprimento ajustado
        """
        try:
            if comprimento_alvo is None:
                comprimento_alvo = self.amostras_alvo
            
            if len(audio) == comprimento_alvo:
                return audio
            
            elif len(audio) > comprimento_alvo:
                # Truncar: manter parte central
                excesso = len(audio) - comprimento_alvo
                inicio = excesso // 2
                fim = inicio + comprimento_alvo
                audio_ajustado = audio[inicio:fim]
                
                logger.debug(f"Áudio truncado: {len(audio)} -> {len(audio_ajustado)} amostras")
                
                return audio_ajustado
            
            else:
                # Padding: adicionar zeros
                falta = comprimento_alvo - len(audio)
                pad_inicio = falta // 2
                pad_fim = falta - pad_inicio
                
                audio_ajustado = np.pad(audio, 
                                       (pad_inicio, pad_fim), 
                                       mode='constant',
                                       constant_values=0)
                
                logger.debug(f"Áudio com padding: {len(audio)} -> {len(audio_ajustado)} amostras")
                
                return audio_ajustado
                
        except Exception as e:
            logger.error(f"Erro ao ajustar comprimento: {e}")
            return audio
    
    # ============================================================================
    # AUMENTO DE DADOS (DATA AUGMENTATION)
    # ============================================================================
    
    def aplicar_aumento_dados(
        self,
        audio: np.ndarray,
        tecnicas: List[str] = None,
        probabilidade: float = 0.5
    ) -> List[np.ndarray]:
        """
        Aplica técnicas de aumento de dados ao áudio.
        
        Args:
            audio: Sinal de áudio original
            tecnicas: Lista de técnicas a aplicar (None para todas)
            probabilidade: Probabilidade de aplicar cada técnica
        
        Returns:
            Lista de áudios aumentados (incluindo o original)
        """
        if tecnicas is None:
            tecnicas = ['ruido', 'pitch', 'tempo', 'reverso', 'ganho']
        
        aumentados = [audio]  # Sempre incluir o original
        
        try:
            for tecnica in tecnicas:
                if np.random.random() > probabilidade:
                    continue
                
                if tecnica == 'ruido' and self.remover_ruido:
                    audio_aug = self.adicionar_ruido_branco(audio)
                    aumentados.append(audio_aug)
                
                elif tecnica == 'pitch':
                    audio_aug = self.modificar_pitch(audio)
                    aumentados.append(audio_aug)
                
                elif tecnica == 'tempo':
                    audio_aug = self.modificar_tempo(audio)
                    aumentados.append(audio_aug)
                
                elif tecnica == 'reverso':
                    audio_aug = self.reverter_audio(audio)
                    aumentados.append(audio_aug)
                
                elif tecnica == 'ganho':
                    audio_aug = self.modificar_ganho(audio)
                    aumentados.append(audio_aug)
            
            logger.debug(f"Aumento de dados aplicado: {len(aumentados)} versões criadas")
            
            return aumentados
            
        except Exception as e:
            logger.error(f"Erro no aumento de dados: {e}")
            return [audio]
    
    def adicionar_ruido_branco(
        self,
        audio: np.ndarray,
        snr_db: float = 20.0
    ) -> np.ndarray:
        """
        Adiciona ruído branco ao áudio com SNR especificada.
        
        Args:
            audio: Sinal de áudio
            snr_db: Relação sinal-ruído em dB
        
        Returns:
            Áudio com ruído adicionado
        """
        try:
            # Calcular potência do sinal
            potencia_sinal = np.mean(audio ** 2)
            
            # Calcular potência do ruído necessária
            snr_linear = 10 ** (snr_db / 10)
            potencia_ruido = potencia_sinal / snr_linear
            
            # Gerar ruído branco
            ruido = np.random.randn(len(audio)) * np.sqrt(potencia_ruido)
            
            # Adicionar ruído ao sinal
            audio_com_ruido = audio + ruido
            
            # Normalizar para evitar clipping
            max_val = np.max(np.abs(audio_com_ruido))
            if max_val > 1.0:
                audio_com_ruido = audio_com_ruido / max_val
            
            logger.debug(f"Ruído branco adicionado: SNR={snr_db}dB")
            
            return audio_com_ruido
            
        except Exception as e:
            logger.error(f"Erro ao adicionar ruído branco: {e}")
            return audio
    
    def modificar_pitch(
        self,
        audio: np.ndarray,
        variacao_semitons: float = 2.0
    ) -> np.ndarray:
        """
        Modifica o pitch (afinação) do áudio.
        
        Args:
            audio: Sinal de áudio
            variacao_semitons: Variação máxima em semitons
        
        Returns:
            Áudio com pitch modificado
        """
        try:
            # Escolher variação aleatória
            semitons = np.random.uniform(-variacao_semitons, variacao_semitons)
            
            # Aplicar mudança de pitch
            audio_pitch = librosa.effects.pitch_shift(
                y=audio,
                sr=self.sr,
                n_steps=semitons
            )
            
            logger.debug(f"Pitch modificado: {semitons:.1f} semitons")
            
            return audio_pitch
            
        except Exception as e:
            logger.error(f"Erro ao modificar pitch: {e}")
            return audio
    
    def modificar_tempo(
        self,
        audio: np.ndarray,
        variacao_taxa: float = 0.2
    ) -> np.ndarray:
        """
        Modifica a velocidade do áudio.
        
        Args:
            audio: Sinal de áudio
            variacao_taxa: Variação máxima na taxa de velocidade
        
        Returns:
            Áudio com velocidade modificada
        """
        try:
            # Escolher taxa aleatória
            taxa = np.random.uniform(1.0 - variacao_taxa, 1.0 + variacao_taxa)
            
            # Aplicar mudança de velocidade
            audio_tempo = librosa.effects.time_stretch(y=audio, rate=taxa)
            
            # Ajustar comprimento se necessário
            if len(audio_tempo) != len(audio):
                audio_tempo = self.ajustar_comprimento(audio_tempo, len(audio))
            
            logger.debug(f"Tempo modificado: taxa={taxa:.2f}")
            
            return audio_tempo
            
        except Exception as e:
            logger.error(f"Erro ao modificar tempo: {e}")
            return audio
    
    def reverter_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Reverte o áudio no tempo.
        
        Args:
            audio: Sinal de áudio
        
        Returns:
            Áudio revertido
        """
        try:
            audio_reverso = np.flip(audio)
            logger.debug("Áudio revertido")
            return audio_reverso
            
        except Exception as e:
            logger.error(f"Erro ao reverter áudio: {e}")
            return audio
    
    def modificar_ganho(
        self,
        audio: np.ndarray,
        variacao_db: float = 6.0
    ) -> np.ndarray:
        """
        Modifica o ganho (volume) do áudio.
        
        Args:
            audio: Sinal de áudio
            variacao_db: Variação máxima em dB
        
        Returns:
            Áudio com ganho modificado
        """
        try:
            # Escolher ganho aleatório em dB
            ganho_db = np.random.uniform(-variacao_db, variacao_db)
            
            # Converter para fator linear
            ganho_linear = 10 ** (ganho_db / 20)
            
            # Aplicar ganho
            audio_ganho = audio * ganho_linear
            
            # Normalizar se necessário
            max_val = np.max(np.abs(audio_ganho))
            if max_val > 1.0:
                audio_ganho = audio_ganho / max_val
            
            logger.debug(f"Ganho modificado: {ganho_db:.1f}dB")
            
            return audio_ganho
            
        except Exception as e:
            logger.error(f"Erro ao modificar ganho: {e}")
            return audio
    
    # ============================================================================
    # PIPELINE COMPLETO DE PRÉ-PROCESSAMENTO
    # ============================================================================
    
    def processar_pipeline(
        self,
        caminho_audio: str,
        extrair_segmento: bool = True,
        aplicar_aumento: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Executa o pipeline completo de pré-processamento.
        
        Args:
            caminho_audio: Caminho para o arquivo de áudio
            extrair_segmento: Se True, extrai apenas segmentos ativos
            aplicar_aumento: Se True, aplica aumento de dados
        
        Returns:
            Áudio(s) processado(s) - lista se aplicar_aumento=True
        """
        logger.info(f"Iniciando processamento: {caminho_audio}")
        
        try:
            # 1. Carregar áudio
            audio, sr_carregado = self.carregar_audio(caminho_audio)
            
            # Verificar taxa de amostragem
            if sr_carregado != self.sr:
                logger.warning(f"SR diferente: {sr_carregado}Hz (esperado {self.sr}Hz)")
                # Redimensionar se necessário
                if len(audio) > 0:
                    audio = librosa.resample(audio, 
                                            orig_sr=sr_carregado, 
                                            target_sr=self.sr)
            
            # 2. Pré-processamento básico
            audio = self.remover_offset_dc(audio)
            audio = self.aplicar_preenfase(audio)
            audio = self.normalizar_amplitude(audio, 'peak')
            
            # 3. Remoção de ruído e filtragem (se habilitado)
            if self.remover_ruido:
                audio = self.remover_ruido_espectral(audio)
                audio = self.aplicar_filtro_wiener(audio)
                audio = self.aplicar_filtro_mediana(audio)
            
            if self.aplicar_filtros:
                audio = self.aplicar_filtro_passa_banda(audio)
            
            # 4. Extrair segmento principal (se habilitado)
            if extrair_segmento:
                audio = self.extrair_segmento_principal(audio)
            else:
                audio = self.ajustar_comprimento(audio)
            
            # 5. Normalização final
            audio = self.normalizar_amplitude(audio, 'peak')
            
            logger.info(f"Processamento concluído: {caminho_audio} "
                       f"shape={audio.shape}, duração={len(audio)/self.sr:.2f}s")
            
            # 6. Aplicar aumento de dados (se habilitado)
            if aplicar_aumento:
                audios_aumentados = self.aplicar_aumento_dados(audio)
                return audios_aumentados
            else:
                return audio
            
        except Exception as e:
            logger.error(f"Erro no pipeline de processamento para {caminho_audio}: {e}")
            # Retornar array vazio em caso de erro
            return np.zeros(self.amostras_alvo)
    
    def processar_lote(
        self,
        caminhos_audios: List[str],
        usar_multithreading: bool = True,
        max_workers: int = 4
    ) -> List[np.ndarray]:
        """
        Processa uma lista de áudios em lote.
        
        Args:
            caminhos_audios: Lista de caminhos para arquivos de áudio
            usar_multithreading: Se True, usa ThreadPoolExecutor
            max_workers: Número máximo de workers para multithreading
        
        Returns:
            Lista de áudios processados
        """
        logger.info(f"Iniciando processamento em lote: {len(caminhos_audios)} áudios")
        
        audios_processados = []
        
        if usar_multithreading and len(caminhos_audios) > 1:
            # Processamento paralelo
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submeter tarefas
                futures = {
                    executor.submit(self.processar_pipeline, caminho): caminho
                    for caminho in caminhos_audios
                }
                
                # Coletar resultados
                for i, future in enumerate(as_completed(futures)):
                    caminho = futures[future]
                    try:
                        resultado = future.result()
                        if isinstance(resultado, list):
                            audios_processados.extend(resultado)
                        else:
                            audios_processados.append(resultado)
                    except Exception as e:
                        logger.error(f"Erro ao processar {caminho}: {e}")
                        # Adicionar array vazio em caso de erro
                        audios_processados.append(np.zeros(self.amostras_alvo))
                    
                    # Log de progresso
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processados {i+1}/{len(caminhos_audios)} áudios")
        
        else:
            # Processamento sequencial
            for i, caminho in enumerate(caminhos_audios):
                try:
                    resultado = self.processar_pipeline(caminho)
                    if isinstance(resultado, list):
                        audios_processados.extend(resultado)
                    else:
                        audios_processados.append(resultado)
                except Exception as e:
                    logger.error(f"Erro ao processar {caminho}: {e}")
                    audios_processados.append(np.zeros(self.amostras_alvo))
                
                # Log de progresso
                if (i + 1) % 10 == 0:
                    logger.info(f"Processados {i+1}/{len(caminhos_audios)} áudios")
        
        logger.info(f"Processamento em lote concluído: "
                   f"{len(audios_processados)} áudios processados")
        
        return audios_processados

# ============================================================================
# FUNÇÕES UTILITÁRIAS ADICIONAIS
# ============================================================================

def analisar_audio(audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
    """
    Analisa estatísticas básicas de um áudio.
    
    Args:
        audio: Sinal de áudio
        sr: Taxa de amostragem
    
    Returns:
        Dicionário com estatísticas do áudio
    """
    stats = {}
    
    try:
        stats['comprimento'] = len(audio)
        stats['duracao'] = len(audio) / sr
        stats['amplitudes'] = {
            'min': float(np.min(audio)),
            'max': float(np.max(audio)),
            'media': float(np.mean(audio)),
            'std': float(np.std(audio))
        }
        
        # Estatísticas de energia
        energia = np.mean(audio ** 2)
        stats['energia'] = float(energia)
        stats['energia_db'] = float(10 * np.log10(energia + 1e-10))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0, 0]
        stats['zcr'] = float(zcr)
        
        # Frequência fundamental (pitch) se aplicável
        try:
            f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
            f0 = f0[f0 > 0]
            if len(f0) > 0:
                stats['f0_media'] = float(np.mean(f0))
                stats['f0_std'] = float(np.std(f0))
        except:
            pass
        
        logger.debug(f"Áudio analisado: {stats['duracao']:.2f}s, "
                    f"amplitude=[{stats['amplitudes']['min']:.3f}, "
                    f"{stats['amplitudes']['max']:.3f}]")
        
        return stats
        
    except Exception as e:
        logger.error(f"Erro na análise do áudio: {e}")
        return {}

# ============================================================================
# FUNÇÃO PRINCIPAL DE TESTE
# ============================================================================

if __name__ == '__main__':
    """
    Teste das funcionalidades do módulo de pré-processamento.
    """
    import tempfile
    
    print("=" * 60)
    print("TESTE DO MÓDULO DE PRÉ-PROCESSAMENTO")
    print("=" * 60)
    
    # Configurar logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Criar áudio de teste sintético
    print("\n1. Criando áudio de teste sintético...")
    sr_teste = 16000
    duracao_teste = 2.0
    t = np.linspace(0, duracao_teste, int(sr_teste * duracao_teste))
    
    # Criar sinal que simula tosse (bursts periódicos)
    audio_teste = np.zeros_like(t)
    for i in range(5):
        inicio = 0.2 + i * 0.3
        fim = inicio + 0.1
        idx = (t >= inicio) & (t <= fim)
        audio_teste[idx] = np.sin(2 * np.pi * 500 * t[idx]) * 0.5
    
    # Adicionar algum ruído
    ruido = np.random.randn(len(t)) * 0.05
    audio_teste += ruido
    
    # Salvar áudio de teste
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        caminho_teste = tmp.name
        sf.write(caminho_teste, audio_teste, sr_teste)
        print(f"Áudio de teste criado: {caminho_teste}")
    
    # Testar pré-processador
    print("\n2. Testando pré-processador...")
    
    preprocessador = PreprocessadorTosse(
        sr=sr_teste,
        duracao_alvo=3.0,
        aplicar_filtros=True,
        remover_ruido=True
    )
    
    # Processar áudio
    print("Processando áudio...")
    audio_processado = preprocessador.processar_pipeline(
        caminho_teste,
        extrair_segmento=True,
        aplicar_aumento=False
    )
    
    print(f"Áudio original: {len(audio_teste)} amostras")
    print(f"Áudio processado: {len(audio_processado)} amostras")
    
    # Testar análise
    print("\n3. Testando análise de áudio...")
    stats = analisar_audio(audio_processado, sr_teste)
    print(f"Estatísticas do áudio:")
    print(f"  Duração: {stats.get('duracao', 0):.2f}s")
    print(f"  Amplitude: [{stats.get('amplitudes', {}).get('min', 0):.3f}, "
          f"{stats.get('amplitudes', {}).get('max', 0):.3f}]")
    print(f"  Energia: {stats.get('energia_db', 0):.1f} dB")
    print(f"  ZCR: {stats.get('zcr', 0):.3f}")
    
    # Testar aumento de dados
    print("\n4. Testando aumento de dados...")
    aumentados = preprocessador.aplicar_aumento_dados(
        audio_processado,
        tecnicas=['ruido', 'pitch', 'tempo'],
        probabilidade=0.7
    )
    print(f"Versões criadas: {len(aumentados)}")
    
    # Testar processamento em lote
    print("\n5. Testando processamento em lote...")
    caminhos_lote = [caminho_teste] * 3
    lote_processado = preprocessador.processar_lote(
        caminhos_lote,
        usar_multithreading=True,
        max_workers=2
    )
    print(f"Áudios processados em lote: {len(lote_processado)}")
    
    # Limpar arquivo temporário
    try:
        os.unlink(caminho_teste)
        print(f"\nArquivo temporário removido: {caminho_teste}")
    except:
        pass
    
    print("\n" + "=" * 60)
    print("TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
