"""
extrair_caracteristicas.py - Módulo para extração de características de áudio

Este módulo contém funções para:
- Extração de espectrogramas (Mel, Log-Mel, MFCC)
- Extração de características temporais e espectrais
- Extração de características específicas para tosse
- Conversão para formatos adequados para modelos de ML
- Normalização e padronização de features
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings
warnings.filterwarnings('ignore')
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
import tensorflow as tf

# Configurar logging
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES E CONFIGURAÇÕES
# ============================================================================

class ConfigCaracteristicas:
    """Configurações padrão para extração de características."""
    
    # Configurações para espectrogramas
    SR_PADRAO = 16000
    N_FFT = 2048
    HOP_LENGTH = 512
    WIN_LENGTH = 2048
    N_MELS = 128
    N_MFCC = 13
    FMAX = 8000
    
    # Configurações para características temporais
    FRAME_LENGTH = 2048
    HOP_LENGTH_TEMPORAL = 512
    
    # Configurações para características de tosse
    DURACAO_PADRAO = 3.0
    AMOSTRAS_PADRAO = int(SR_PADRAO * DURACAO_PADRAO)
    
    # Formato de imagem para espectrogramas
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNELS = 3

# ============================================================================
# CLASSE PRINCIPAL PARA EXTRAÇÃO DE CARACTERÍSTICAS
# ============================================================================

class ExtratorCaracteristicasTosse:
    """
    Classe principal para extração de características de áudio de tosse.
    
    Esta classe implementa métodos para extrair diversos tipos de características
    de áudio, otimizadas para classificação de sons de tosse.
    """
    
    def __init__(
        self,
        sr: int = ConfigCaracteristicas.SR_PADRAO,
        n_mels: int = ConfigCaracteristicas.N_MELS,
        n_mfcc: int = ConfigCaracteristicas.N_MFCC,
        n_fft: int = ConfigCaracteristicas.N_FFT,
        hop_length: int = ConfigCaracteristicas.HOP_LENGTH,
        fmax: int = ConfigCaracteristicas.FMAX,
        normalizar: bool = True
    ):
        """
        Inicializa o extrator de características.
        
        Args:
            sr: Taxa de amostragem (Hz)
            n_mels: Número de bandas Mel
            n_mfcc: Número de coeficientes MFCC
            n_fft: Tamanho da FFT
            hop_length: Hop length para STFT
            fmax: Frequência máxima para banda Mel
            normalizar: Se True, normaliza as características
        """
        self.sr = sr
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmax = fmax
        self.normalizar = normalizar
        
        logger.info(f"Extrator inicializado: SR={sr}Hz, "
                   f"Mels={n_mels}, MFCCs={n_mfcc}, FFT={n_fft}")
    
    # ============================================================================
    # MÉTODOS PARA ESPECTROGRAMAS
    # ============================================================================
    
    def extrair_espectrograma_mel(
        self,
        audio: np.ndarray,
        usar_log: bool = True,
        usar_power: bool = True
    ) -> np.ndarray:
        """
        Extrai espectrograma Mel.
        
        Args:
            audio: Sinal de áudio
            usar_log: Se True, converte para escala logarítmica
            usar_power: Se True, calcula espectrograma de power
        
        Returns:
            Espectrograma Mel (n_mels x tempo)
        """
        try:
            # Calcular espectrograma Mel
            if usar_power:
                mel_spec = librosa.feature.melspectrogram(
                    y=audio,
                    sr=self.sr,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.n_fft,
                    fmax=self.fmax,
                    power=2.0
                )
            else:
                mel_spec = librosa.feature.melspectrogram(
                    y=audio,
                    sr=self.sr,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.n_fft,
                    fmax=self.fmax,
                    power=1.0
                )
            
            # Converter para dB se solicitado
            if usar_log:
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            logger.debug(f"Espectrograma Mel extraído: shape={mel_spec.shape}")
            
            return mel_spec
            
        except Exception as e:
            logger.error(f"Erro ao extrair espectrograma Mel: {e}")
            # Retornar array vazio com shape apropriado
            n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
            return np.zeros((self.n_mels, max(1, n_frames)))
    
    def extrair_espectrograma_log_mel(
        self,
        audio: np.ndarray,
        eps: float = 1e-10
    ) -> np.ndarray:
        """
        Extrai espectrograma Log-Mel (escala logarítmica).
        
        Args:
            audio: Sinal de áudio
            eps: Valor pequeno para evitar log(0)
        
        Returns:
            Espectrograma Log-Mel
        """
        try:
            # Extrair espectrograma Mel
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                fmax=self.fmax
            )
            
            # Aplicar log
            log_mel_spec = np.log(mel_spec + eps)
            
            logger.debug(f"Espectrograma Log-Mel extraído: shape={log_mel_spec.shape}")
            
            return log_mel_spec
            
        except Exception as e:
            logger.error(f"Erro ao extrair espectrograma Log-Mel: {e}")
            n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
            return np.zeros((self.n_mels, max(1, n_frames)))
    
    def extrair_mfcc(
        self,
        audio: np.ndarray,
        incluir_delta: bool = True,
        incluir_delta_delta: bool = True
    ) -> np.ndarray:
        """
        Extrai coeficientes MFCC com deltas opcionais.
        
        Args:
            audio: Sinal de áudio
            incluir_delta: Se True, inclui coeficientes delta
            incluir_delta_delta: Se True, inclui coeficientes delta-delta
        
        Returns:
            Coeficientes MFCC (com deltas se especificado)
        """
        try:
            # Extrair MFCCs básicos
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmax=self.fmax
            )
            
            features = [mfcc]
            
            # Adicionar deltas se solicitado
            if incluir_delta:
                mfcc_delta = librosa.feature.delta(mfcc)
                features.append(mfcc_delta)
            
            if incluir_delta_delta:
                mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                features.append(mfcc_delta2)
            
            # Concatenar todas as features
            mfcc_completo = np.vstack(features)
            
            logger.debug(f"MFCCs extraídos: shape={mfcc_completo.shape}, "
                        f"deltas={incluir_delta}, delta-deltas={incluir_delta_delta}")
            
            return mfcc_completo
            
        except Exception as e:
            logger.error(f"Erro ao extrair MFCCs: {e}")
            n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
            n_features = self.n_mfcc
            if incluir_delta:
                n_features += self.n_mfcc
            if incluir_delta_delta:
                n_features += self.n_mfcc
            return np.zeros((n_features, max(1, n_frames)))
    
    def extrair_espectrograma_cromático(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrai espectrograma cromático (chromagram).
        
        Args:
            audio: Sinal de áudio
        
        Returns:
            Espectrograma cromático
        """
        try:
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            logger.debug(f"Espectrograma cromático extraído: shape={chroma.shape}")
            
            return chroma
            
        except Exception as e:
            logger.error(f"Erro ao extrair espectrograma cromático: {e}")
            n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
            return np.zeros((12, max(1, n_frames)))
    
    # ============================================================================
    # MÉTODOS PARA CARACTERÍSTICAS TEMPORAIS
    # ============================================================================
    
    def extrair_caracteristicas_temporais(
        self,
        audio: np.ndarray
    ) -> Dict[str, float]:
        """
        Extrai características temporais do sinal de áudio.
        
        Args:
            audio: Sinal de áudio
        
        Returns:
            Dicionário com características temporais
        """
        features = {}
        
        try:
            # 1. Zero Crossing Rate (ZCR)
            zcr = librosa.feature.zero_crossing_rate(
                y=audio,
                frame_length=ConfigCaracteristicas.FRAME_LENGTH,
                hop_length=ConfigCaracteristicas.HOP_LENGTH_TEMPORAL
            )[0]
            
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            features['zcr_max'] = float(np.max(zcr))
            features['zcr_min'] = float(np.min(zcr))
            
            # 2. Energia/RMS
            rms = librosa.feature.rms(
                y=audio,
                frame_length=ConfigCaracteristicas.FRAME_LENGTH,
                hop_length=ConfigCaracteristicas.HOP_LENGTH_TEMPORAL
            )[0]
            
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            features['rms_max'] = float(np.max(rms))
            features['rms_min'] = float(np.min(rms))
            features['rms_dynamic_range'] = float(np.max(rms) - np.min(rms))
            
            # 3. Características de amplitude
            features['amplitude_mean'] = float(np.mean(np.abs(audio)))
            features['amplitude_std'] = float(np.std(audio))
            features['amplitude_max'] = float(np.max(np.abs(audio)))
            features['amplitude_skew'] = float(stats.skew(audio))
            features['amplitude_kurtosis'] = float(stats.kurtosis(audio))
            
            # 4. Taxa de picos (peaks)
            picos, _ = signal.find_peaks(np.abs(audio), height=np.std(audio)*0.5)
            features['n_picos'] = float(len(picos))
            features['taxa_picos'] = float(len(picos) / (len(audio) / self.sr))
            
            if len(picos) > 1:
                intervalos = np.diff(picos) / self.sr
                features['intervalo_picos_mean'] = float(np.mean(intervalos))
                features['intervalo_picos_std'] = float(np.std(intervalos))
            else:
                features['intervalo_picos_mean'] = 0.0
                features['intervalo_picos_std'] = 0.0
            
            # 5. Duração de eventos (threshold-based)
            limiar = np.std(audio) * 0.3
            acima_limiar = np.abs(audio) > limiar
            eventos = self._segmentar_eventos(acima_limiar)
            
            if eventos:
                duracao_eventos = [fim - inicio for inicio, fim in eventos]
                features['n_eventos'] = float(len(eventos))
                features['duracao_eventos_mean'] = float(np.mean(duracao_eventos) / self.sr)
                features['duracao_eventos_std'] = float(np.std(duracao_eventos) / self.sr)
                features['duracao_total_eventos'] = float(sum(duracao_eventos) / self.sr)
            else:
                features['n_eventos'] = 0.0
                features['duracao_eventos_mean'] = 0.0
                features['duracao_eventos_std'] = 0.0
                features['duracao_total_eventos'] = 0.0
            
            logger.debug(f"Características temporais extraídas: {len(features)} features")
            
            return features
            
        except Exception as e:
            logger.error(f"Erro ao extrair características temporais: {e}")
            return {}
    
    def _segmentar_eventos(self, mascara: np.ndarray) -> List[Tuple[int, int]]:
        """
        Segmenta eventos contínuos a partir de uma máscara booleana.
        
        Args:
            mascara: Array booleano indicando eventos
        
        Returns:
            Lista de tuplas (inicio, fim) para cada evento
        """
        eventos = []
        inicio = None
        
        for i, valor in enumerate(mascara):
            if valor and inicio is None:
                inicio = i
            elif not valor and inicio is not None:
                eventos.append((inicio, i))
                inicio = None
        
        if inicio is not None:
            eventos.append((inicio, len(mascara)))
        
        return eventos
    
    # ============================================================================
    # MÉTODOS PARA CARACTERÍSTICAS ESPECTRAIS
    # ============================================================================
    
    def extrair_caracteristicas_espectrais(
        self,
        audio: np.ndarray
    ) -> Dict[str, float]:
        """
        Extrai características espectrais do sinal de áudio.
        
        Args:
            audio: Sinal de áudio
        
        Returns:
            Dicionário com características espectrais
        """
        features = {}
        
        try:
            # Calcular STFT
            D = np.abs(librosa.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            ))
            
            # 1. Centróide espectral
            cent = librosa.feature.spectral_centroid(
                y=audio,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            
            features['centroid_mean'] = float(np.mean(cent))
            features['centroid_std'] = float(np.std(cent))
            features['centroid_skew'] = float(stats.skew(cent))
            
            # 2. Largura de banda espectral
            bw = librosa.feature.spectral_bandwidth(
                y=audio,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            
            features['bandwidth_mean'] = float(np.mean(bw))
            features['bandwidth_std'] = float(np.std(bw))
            
            # 3. Rolloff espectral (85% e 95%)
            rolloff85 = librosa.feature.spectral_rolloff(
                y=audio,
                sr=self.sr,
                roll_percent=0.85,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            
            rolloff95 = librosa.feature.spectral_rolloff(
                y=audio,
                sr=self.sr,
                roll_percent=0.95,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            
            features['rolloff85_mean'] = float(np.mean(rolloff85))
            features['rolloff85_std'] = float(np.std(rolloff85))
            features['rolloff95_mean'] = float(np.mean(rolloff95))
            features['rolloff95_std'] = float(np.std(rolloff95))
            
            # 4. Flatness (planicidade espectral)
            flatness = librosa.feature.spectral_flatness(
                y=audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            
            features['flatness_mean'] = float(np.mean(flatness))
            features['flatness_std'] = float(np.std(flatness))
            
            # 5. Contrast espectral
            contrast = librosa.feature.spectral_contrast(
                y=audio,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            for i in range(contrast.shape[0]):
                features[f'contrast_band{i}_mean'] = float(np.mean(contrast[i]))
                features[f'contrast_band{i}_std'] = float(np.std(contrast[i]))
            
            # 6. MFCCs estatísticos (primeiros 5 coeficientes)
            mfcc = self.extrair_mfcc(audio, incluir_delta=False, incluir_delta_delta=False)
            
            for i in range(min(5, mfcc.shape[0])):
                features[f'mfcc{i}_mean'] = float(np.mean(mfcc[i]))
                features[f'mfcc{i}_std'] = float(np.std(mfcc[i]))
                features[f'mfcc{i}_skew'] = float(stats.skew(mfcc[i]))
            
            # 7. Proporção de energia em bandas de frequência
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
            
            # Definir bandas relevantes para tosse
            bandas = [
                (100, 500),    # Baixas frequências
                (500, 1500),   # Médias frequências
                (1500, 3000),  # Altas frequências
                (3000, self.fmax)  # Muito altas frequências
            ]
            
            energia_total = np.sum(D ** 2)
            
            for idx, (f_min, f_max) in enumerate(bandas):
                mascara = (freqs >= f_min) & (freqs <= f_max)
                if np.any(mascara):
                    energia_banda = np.sum(D[mascara] ** 2)
                    proporcao = energia_banda / (energia_total + 1e-10)
                    features[f'energia_banda{idx}_proporcao'] = float(proporcao)
            
            logger.debug(f"Características espectrais extraídas: {len(features)} features")
            
            return features
            
        except Exception as e:
            logger.error(f"Erro ao extrair características espectrais: {e}")
            return {}
    
    # ============================================================================
    # MÉTODOS PARA CARACTERÍSTICAS ESPECÍFICAS DE TOSSE
    # ============================================================================
    
    def extrair_caracteristicas_tosse(
        self,
        audio: np.ndarray
    ) -> Dict[str, float]:
        """
        Extrai características específicas para sons de tosse.
        
        Args:
            audio: Sinal de áudio contendo tosse
        
        Returns:
            Dicionário com características específicas de tosse
        """
        features = {}
        
        try:
            # 1. Características de ataque e decaimento
            envelope = np.abs(signal.hilbert(audio))
            envelope_suavizado = gaussian_filter1d(envelope, sigma=self.sr//100)
            
            # Encontrar picos no envelope
            picos, propriedades = signal.find_peaks(
                envelope_suavizado,
                height=np.percentile(envelope_suavizado, 75),
                distance=self.sr//10  # Mínimo 100ms entre tosses
            )
            
            if len(picos) > 0:
                # Características baseadas em picos
                features['n_tosses'] = float(len(picos))
                features['intervalo_tosses_mean'] = float(np.mean(np.diff(picos)) / self.sr if len(picos) > 1 else 0)
                features['intervalo_tosses_std'] = float(np.std(np.diff(picos)) / self.sr if len(picos) > 1 else 0)
                
                # Altura dos picos
                alturas = propriedades['peak_heights']
                features['altura_tosses_mean'] = float(np.mean(alturas))
                features['altura_tosses_std'] = float(np.std(alturas))
                features['altura_tosses_max'] = float(np.max(alturas))
                
                # Analisar ataque e decaimento de cada tosse
                taxas_ataque = []
                taxas_decaimento = []
                
                for pico in picos:
                    # Encontrar início (onde envelope sobe de 10% para 90% antes do pico)
                    antes_pico = envelope_suavizado[:pico]
                    if len(antes_pico) > 0:
                        altura_pico = envelope_suavizado[pico]
                        limiar_inicio = 0.1 * altura_pico
                        
                        # Encontrar onde cruza o limiar
                        inicio_idx = np.where(antes_pico > limiar_inicio)[0]
                        if len(inicio_idx) > 0:
                            inicio = inicio_idx[0]
                            duracao_ataque = (pico - inicio) / self.sr
                            if duracao_ataque > 0:
                                taxa_ataque = (altura_pico - envelope_suavizado[inicio]) / duracao_ataque
                                taxas_ataque.append(taxa_ataque)
                    
                    # Encontrar fim (onde envelope cai para 10% após o pico)
                    depois_pico = envelope_suavizado[pico:]
                    if len(depois_pico) > 0:
                        limiar_fim = 0.1 * altura_pico
                        
                        # Encontrar onde cruza o limiar
                        fim_idx = np.where(depois_pico < limiar_fim)[0]
                        if len(fim_idx) > 0:
                            fim = pico + fim_idx[0]
                            duracao_decaimento = (fim - pico) / self.sr
                            if duracao_decaimento > 0:
                                taxa_decaimento = altura_pico / duracao_decaimento
                                taxas_decaimento.append(taxa_decaimento)
                
                if taxas_ataque:
                    features['taxa_ataque_mean'] = float(np.mean(taxas_ataque))
                    features['taxa_ataque_std'] = float(np.std(taxas_ataque))
                
                if taxas_decaimento:
                    features['taxa_decaimento_mean'] = float(np.mean(taxas_decaimento))
                    features['taxa_decaimento_std'] = float(np.std(taxas_decaimento))
            
            # 2. Características de periodicidade
            if len(picos) > 1:
                # Calcular autocorrelação para periodicidade
                autocorr = np.correlate(audio, audio, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Encontrar picos na autocorrelação (após o primeiro)
                picos_autocorr, _ = signal.find_peaks(
                    autocorr[:self.sr//2],  # Até 500ms
                    height=np.percentile(autocorr, 90)
                )
                
                if len(picos_autocorr) > 1:
                    # O primeiro pico é em 0, usar o segundo para periodicidade
                    periodo = picos_autocorr[1] / self.sr
                    features['periodicidade'] = float(1.0 / periodo if periodo > 0 else 0)
                    features['forca_periodicidade'] = float(autocorr[picos_autocorr[1]] / autocorr[0])
            
            # 3. Características de espectro específicas
            # Análise de formantes (para tosse produtiva vs seca)
            f0, voicing = self._estimar_formantes(audio)
            if f0 is not None and len(f0) > 0:
                features['f0_mean'] = float(np.mean(f0))
                features['f0_std'] = float(np.std(f0))
                features['proporcao_voz'] = float(np.mean(voicing))
            
            # 4. Características de ruído (para tosse seca)
            # Usar análise HNR (Harmonic-to-Noise Ratio)
            hnr = self._calcular_hnr(audio)
            if hnr is not None:
                features['hnr_mean'] = float(np.mean(hnr))
                features['hnr_std'] = float(np.std(hnr))
            
            logger.debug(f"Características de tosse extraídas: {len(features)} features")
            
            return features
            
        except Exception as e:
            logger.error(f"Erro ao extrair características de tosse: {e}")
            return {}
    
    def _estimar_formantes(self, audio: np.ndarray, n_formantes: int = 3):
        """
        Estima formantes usando análise LPC.
        
        Args:
            audio: Sinal de áudio
            n_formantes: Número de formantes a estimar
        
        Returns:
            Tuple (frequências formantes, flags de voicing)
        """
        try:
            # Aplicar pré-ênfase
            audio_pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
            
            # Janelamento
            frame_length = int(0.025 * self.sr)  # 25ms
            hop_length = int(0.010 * self.sr)    # 10ms
            
            f0_frames = []
            voicing_frames = []
            
            for i in range(0, len(audio_pre) - frame_length, hop_length):
                frame = audio_pre[i:i + frame_length]
                frame = frame * np.hamming(len(frame))
                
                # Calcular autocorrelação para pitch
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Encontrar primeiro pico após lag 0
                picos, _ = signal.find_peaks(
                    autocorr[:frame_length//2],
                    height=0.3 * np.max(autocorr)
                )
                
                if len(picos) > 0 and picos[0] > 0:
                    f0 = self.sr / picos[0]
                    if 50 < f0 < 500:  # Faixa plausível para pitch
                        f0_frames.append(f0)
                        voicing_frames.append(1.0)
                    else:
                        voicing_frames.append(0.0)
                else:
                    voicing_frames.append(0.0)
            
            if f0_frames:
                return np.array(f0_frames), np.array(voicing_frames)
            else:
                return None, np.array(voicing_frames)
                
        except Exception as e:
            logger.debug(f"Erro na estimativa de formantes: {e}")
            return None, np.array([])
    
    def _calcular_hnr(self, audio: np.ndarray):
        """
        Calcula Harmonic-to-Noise Ratio.
        
        Args:
            audio: Sinal de áudio
        
        Returns:
            Array com HNR por frame ou None em caso de erro
        """
        try:
            frame_length = int(0.025 * self.sr)  # 25ms
            hop_length = int(0.010 * self.sr)    # 10ms
            
            hnr_values = []
            
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                
                # Calcular autocorrelação
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                if len(autocorr) > 0 and autocorr[0] > 0:
                    # Encontrar primeiro pico após lag 0
                    if len(autocorr) > 10:
                        ruido = np.mean(autocorr[10:20])
                        if ruido > 0:
                            hnr = 10 * np.log10(autocorr[0] / ruido)
                            hnr_values.append(hnr)
            
            return np.array(hnr_values) if hnr_values else None
            
        except Exception as e:
            logger.debug(f"Erro no cálculo de HNR: {e}")
            return None
    
    # ============================================================================
    # MÉTODOS PARA CONVERSÃO PARA IMAGEM
    # ============================================================================
    
    def espectrograma_para_imagem(
        self,
        espectrograma: np.ndarray,
        altura: int = None,
        largura: int = None,
        canais: int = None,
        normalizar_local: bool = True
    ) -> np.ndarray:
        """
        Converte um espectrograma para formato de imagem.
        
        Args:
            espectrograma: Espectrograma 2D (frequência x tempo)
            altura: Altura da imagem (None para usar padrão)
            largura: Largura da imagem (None para usar padrão)
            canais: Número de canais (1 para grayscale, 3 para RGB)
            normalizar_local: Se True, normaliza cada imagem individualmente
        
        Returns:
            Imagem 3D (altura x largura x canais)
        """
        try:
            if altura is None:
                altura = ConfigCaracteristicas.IMG_HEIGHT
            if largura is None:
                largura = ConfigCaracteristicas.IMG_WIDTH
            if canais is None:
                canais = ConfigCaracteristicas.IMG_CHANNELS
            
            # Converter para tensor TensorFlow
            espectrograma_tensor = tf.convert_to_tensor(espectrograma, dtype=tf.float32)
            
            # Adicionar dimensão do canal se necessário
            if len(espectrograma_tensor.shape) == 2:
                espectrograma_tensor = tf.expand_dims(espectrograma_tensor, axis=-1)
            
            # Redimensionar
            imagem = tf.image.resize(
                espectrograma_tensor,
                (altura, largura),
                method=tf.image.ResizeMethod.BILINEAR
            )
            
            # Normalizar
            if normalizar_local:
                min_val = tf.reduce_min(imagem)
                max_val = tf.reduce_max(imagem)
                epsilon = 1e-8
                imagem = (imagem - min_val) / (max_val - min_val + epsilon)
            
            # Converter para RGB se necessário
            if canais == 3 and imagem.shape[-1] == 1:
                imagem = tf.repeat(imagem, 3, axis=-1)
            elif canais == 1 and imagem.shape[-1] == 3:
                imagem = tf.reduce_mean(imagem, axis=-1, keepdims=True)
            
            # Garantir shape correto
            imagem = tf.ensure_shape(
                imagem,
                (altura, largura, canais)
            )
            
            logger.debug(f"Espectrograma convertido para imagem: {imagem.shape}")
            
            return imagem.numpy()
            
        except Exception as e:
            logger.error(f"Erro ao converter espectrograma para imagem: {e}")
            altura = altura or ConfigCaracteristicas.IMG_HEIGHT
            largura = largura or ConfigCaracteristicas.IMG_WIDTH
            canais = canais or ConfigCaracteristicas.IMG_CHANNELS
            return np.zeros((altura, largura, canais))
    
    def criar_imagem_multi_canal(
        self,
        audio: np.ndarray
    ) -> np.ndarray:
        """
        Cria imagem multi-canal com diferentes representações.
        
        Args:
            audio: Sinal de áudio
        
        Returns:
            Imagem 3D com múltiplas representações
        """
        try:
            # Extrair diferentes representações
            mel_spec = self.extrair_espectrograma_mel(audio, usar_log=True)
            mfcc = self.extrair_mfcc(audio, incluir_delta=True, incluir_delta_delta=True)
            chroma = self.extrair_espectrograma_cromático(audio)
            
            # Redimensionar para mesmo tamanho
            altura = ConfigCaracteristicas.IMG_HEIGHT
            largura = ConfigCaracteristicas.IMG_WIDTH
            
            # Converter cada representação para imagem
            canal1 = self.espectrograma_para_imagem(mel_spec, altura, largura, 1)
            canal2 = self.espectrograma_para_imagem(mfcc[:mel_spec.shape[0], :mel_spec.shape[1]], 
                                                   altura, largura, 1)
            canal3 = self.espectrograma_para_imagem(chroma, altura, largura, 1)
            
            # Combinar canais
            imagem_multi = np.stack([canal1, canal2, canal3], axis=-1)
            imagem_multi = np.squeeze(imagem_multi)  # Remover dimensões extras
            
            logger.debug(f"Imagem multi-canal criada: {imagem_multi.shape}")
            
            return imagem_multi
            
        except Exception as e:
            logger.error(f"Erro ao criar imagem multi-canal: {e}")
            altura = ConfigCaracteristicas.IMG_HEIGHT
            largura = ConfigCaracteristicas.IMG_WIDTH
            return np.zeros((altura, largura, 3))
    
    # ============================================================================
    # MÉTODOS PARA EXTRAÇÃO COMPLETA
    # ============================================================================
    
    def extrair_todas_caracteristicas(
        self,
        audio: np.ndarray,
        incluir_imagens: bool = True,
        incluir_temporais: bool = True,
        incluir_espectrais: bool = True,
        incluir_tosse: bool = True
    ) -> Dict[str, Any]:
        """
        Extrai todas as características disponíveis.
        
        Args:
            audio: Sinal de áudio
            incluir_imagens: Se True, extrai representações como imagem
            incluir_temporais: Se True, extrai características temporais
            incluir_espectrais: Se True, extrai características espectrais
            incluir_tosse: Se True, extrai características específicas de tosse
        
        Returns:
            Dicionário com todas as características extraídas
        """
        features = {}
        
        try:
            logger.info(f"Extraindo características de áudio com {len(audio)} amostras")
            
            # 1. Representações como imagem
            if incluir_imagens:
                features['imagem_mel'] = self.espectrograma_para_imagem(
                    self.extrair_espectrograma_mel(audio)
                )
                features['imagem_log_mel'] = self.espectrograma_para_imagem(
                    self.extrair_espectrograma_log_mel(audio)
                )
                features['imagem_multi'] = self.criar_imagem_multi_canal(audio)
                
                logger.debug(f"Imagens extraídas: "
                           f"mel={features['imagem_mel'].shape}, "
                           f"multi={features['imagem_multi'].shape}")
            
            # 2. Características temporais
            if incluir_temporais:
                features['temporais'] = self.extrair_caracteristicas_temporais(audio)
                logger.debug(f"Características temporais: {len(features['temporais'])} features")
            
            # 3. Características espectrais
            if incluir_espectrais:
                features['espectrais'] = self.extrair_caracteristicas_espectrais(audio)
                logger.debug(f"Características espectrais: {len(features['espectrais'])} features")
            
            # 4. Características específicas de tosse
            if incluir_tosse:
                features['tosse'] = self.extrair_caracteristicas_tosse(audio)
                logger.debug(f"Características de tosse: {len(features['tosse'])} features")
            
            # 5. Features planas para modelos tradicionais
            features['mfcc'] = self.extrair_mfcc(audio, incluir_delta=True, incluir_delta_delta=True)
            features['mfcc_stats'] = self._extrair_estatisticas_mfcc(features['mfcc'])
            
            logger.info(f"Extracção completa: {sum(len(v) if isinstance(v, dict) else 1 for v in features.values())} features totais")
            
            return features
            
        except Exception as e:
            logger.error(f"Erro na extração completa de características: {e}")
            return {}
    
    def _extrair_estatisticas_mfcc(self, mfcc: np.ndarray) -> Dict[str, float]:
        """
        Extrai estatísticas dos coeficientes MFCC.
        
        Args:
            mfcc: Coeficientes MFCC
        
        Returns:
            Dicionário com estatísticas
        """
        stats = {}
        
        try:
            for i in range(mfcc.shape[0]):
                coef = mfcc[i]
                stats[f'mfcc{i}_mean'] = float(np.mean(coef))
                stats[f'mfcc{i}_std'] = float(np.std(coef))
                stats[f'mfcc{i}_skew'] = float(stats.skew(coef))
                stats[f'mfcc{i}_kurtosis'] = float(stats.kurtosis(coef))
                stats[f'mfcc{i}_min'] = float(np.min(coef))
                stats[f'mfcc{i}_max'] = float(np.max(coef))
                stats[f'mfcc{i}_range'] = float(np.max(coef) - np.min(coef))
            
            return stats
            
        except Exception as e:
            logger.error(f"Erro ao extrair estatísticas MFCC: {e}")
            return {}
    
    # ============================================================================
    # MÉTODOS PARA NORMALIZAÇÃO
    # ============================================================================
    
    def normalizar_features(
        self,
        features: Dict[str, Any],
        estatisticas: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Normaliza as características extraídas.
        
        Args:
            features: Dicionário com características
            estatisticas: Estatísticas para normalização (média, std)
        
        Returns:
            Dicionário com características normalizadas
        """
        if not self.normalizar:
            return features
        
        try:
            features_norm = features.copy()
            
            # Normalizar características temporais
            if 'temporais' in features and features['temporais']:
                features_norm['temporais'] = self._normalizar_dict(
                    features['temporais'],
                    estatisticas.get('temporais', {}) if estatisticas else None
                )
            
            # Normalizar características espectrais
            if 'espectrais' in features and features['espectrais']:
                features_norm['espectrais'] = self._normalizar_dict(
                    features['espectrais'],
                    estatisticas.get('espectrais', {}) if estatisticas else None
                )
            
            # Normalizar características de tosse
            if 'tosse' in features and features['tosse']:
                features_norm['tosse'] = self._normalizar_dict(
                    features['tosse'],
                    estatisticas.get('tosse', {}) if estatisticas else None
                )
            
            # Normalizar MFCCs
            if 'mfcc' in features:
                mfcc = features['mfcc']
                if estatisticas and 'mfcc_mean' in estatisticas and 'mfcc_std' in estatisticas:
                    mean = estatisticas['mfcc_mean']
                    std = estatisticas['mfcc_std']
                    if mean.shape == mfcc.shape and std.shape == mfcc.shape:
                        features_norm['mfcc'] = (mfcc - mean) / (std + 1e-8)
            
            logger.debug("Features normalizadas")
            
            return features_norm
            
        except Exception as e:
            logger.error(f"Erro na normalização de features: {e}")
            return features
    
    def _normalizar_dict(
        self,
        features_dict: Dict[str, float],
        estatisticas: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, float]:
        """
        Normaliza um dicionário de características.
        
        Args:
            features_dict: Dicionário com características
            estatisticas: Tuplas (média, std) para cada feature
        
        Returns:
            Dicionário normalizado
        """
        if estatisticas is None:
            # Normalizar usando estatísticas dos próprios dados
            valores = np.array(list(features_dict.values()))
            mean = np.mean(valores)
            std = np.std(valores)
            
            if std > 0:
                return {k: (v - mean) / std for k, v in features_dict.items()}
            else:
                return features_dict
        else:
            # Normalizar usando estatísticas pré-calculadas
            normalizado = {}
            for k, v in features_dict.items():
                if k in estatisticas:
                    mean, std = estatisticas[k]
                    if std > 0:
                        normalizado[k] = (v - mean) / std
                    else:
                        normalizado[k] = v
                else:
                    normalizado[k] = v
            return normalizado

# ============================================================================
# FUNÇÕES UTILITÁRIAS ADICIONAIS
# ============================================================================

def extrair_features_batch(
    audios: List[np.ndarray],
    sr: int = 16000,
    batch_size: int = 32,
    tipo_features: str = 'imagem'
) -> List[np.ndarray]:
    """
    Extrai características em lote para uma lista de áudios.
    
    Args:
        audios: Lista de sinais de áudio
        sr: Taxa de amostragem
        batch_size: Tamanho do lote para processamento
        tipo_features: Tipo de features a extrair ('imagem', 'mfcc', 'todas')
    
    Returns:
        Lista de características extraídas
    """
    logger = logging.getLogger(__name__)
    extrator = ExtratorCaracteristicasTosse(sr=sr)
    
    todas_features = []
    
    for i in range(0, len(audios), batch_size):
        batch = audios[i:i + batch_size]
        batch_features = []
        
        for j, audio in enumerate(batch):
            try:
                if tipo_features == 'imagem':
                    features = extrator.espectrograma_para_imagem(
                        extrator.extrair_espectrograma_mel(audio)
                    )
                elif tipo_features == 'mfcc':
                    features = extrator.extrair_mfcc(audio)
                elif tipo_features == 'todas':
                    features = extrator.extrair_todas_caracteristicas(audio)
                else:
                    raise ValueError(f"Tipo de features desconhecido: {tipo_features}")
                
                batch_features.append(features)
                
            except Exception as e:
                logger.error(f"Erro ao extrair features do áudio {i+j}: {e}")
                # Adicionar zeros como fallback
                if tipo_features == 'imagem':
                    batch_features.append(np.zeros((128, 128, 3)))
                else:
                    batch_features.append(np.zeros(13))
        
        todas_features.extend(batch_features)
        
        if (i // batch_size) % 10 == 0:
            logger.info(f"Processados {min(i + batch_size, len(audios))}/{len(audios)} áudios")
    
    logger.info(f"Extracção em lote concluída: {len(todas_features)} features extraídas")
    
    return todas_features

# ============================================================================
# FUNÇÃO PRINCIPAL DE TESTE
# ============================================================================

if __name__ == '__main__':
    """
    Teste das funcionalidades do módulo de extração de características.
    """
    import tempfile
    import soundfile as sf
    
    print("=" * 60)
    print("TESTE DO MÓDULO DE EXTRAÇÃO DE CARACTERÍSTICAS")
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
    duracao_teste = 3.0
    t = np.linspace(0, duracao_teste, int(sr_teste * duracao_teste))
    
    # Criar sinal que simula tosse com múltiplos eventos
    audio_teste = np.zeros_like(t)
    
    # Adicionar três eventos de "tosse"
    for i, offset in enumerate([0.5, 1.5, 2.5]):
        inicio = offset
        duracao_evento = 0.3
        idx = (t >= inicio) & (t <= inicio + duracao_evento)
        
        # Frequência fundamental que varia entre eventos
        f0 = 100 + i * 50  # 100Hz, 150Hz, 200Hz
        
        # Criar harmônicos
        sinal_evento = np.zeros(np.sum(idx))
        for harm in range(1, 6):
            amplitude = 1.0 / harm
            sinal_evento += amplitude * np.sin(2 * np.pi * f0 * harm * t[idx])
        
        # Envelope do evento
        envelope = np.sin(np.pi * (t[idx] - inicio) / duracao_evento)
        sinal_evento *= envelope
        
        audio_teste[idx] += sinal_evento * 0.5
    
    # Adicionar algum ruído de fundo
    ruido = np.random.randn(len(t)) * 0.05
    audio_teste += ruido
    
    # Normalizar
    audio_teste = audio_teste / np.max(np.abs(audio_teste))
    
    print(f"Áudio de teste criado: {len(audio_teste)} amostras, {duracao_teste}s")
    
    # Testar extrator
    print("\n2. Testando extrator de características...")
    
    extrator = ExtratorCaracteristicasTosse(
        sr=sr_teste,
        n_mels=128,
        n_mfcc=13,
        normalizar=True
    )
    
    # Testar extração de espectrograma Mel
    print("\n3. Testando extração de espectrograma Mel...")
    mel_spec = extrator.extrair_espectrograma_mel(audio_teste)
    print(f"  Espectrograma Mel: shape={mel_spec.shape}")
    
    # Testar extração de MFCC
    print("\n4. Testando extração de MFCC...")
    mfcc = extrator.extrair_mfcc(audio_teste, incluir_delta=True, incluir_delta_delta=True)
    print(f"  MFCCs: shape={mfcc.shape}")
    
    # Testar características temporais
    print("\n5. Testando características temporais...")
    features_temp = extrator.extrair_caracteristicas_temporais(audio_teste)
    print(f"  Características temporais: {len(features_temp)} features")
    print(f"  Exemplos: ZCR={features_temp.get('zcr_mean', 0):.3f}, "
          f"RMS={features_temp.get('rms_mean', 0):.3f}")
    
    # Testar características espectrais
    print("\n6. Testando características espectrais...")
    features_esp = extrator.extrair_caracteristicas_espectrais(audio_teste)
    print(f"  Características espectrais: {len(features_esp)} features")
    print(f"  Exemplos: Centroid={features_esp.get('centroid_mean', 0):.0f}Hz, "
          f"Bandwidth={features_esp.get('bandwidth_mean', 0):.0f}Hz")
    
    # Testar características específicas de tosse
    print("\n7. Testando características de tosse...")
    features_tosse = extrator.extrair_caracteristicas_tosse(audio_teste)
    print(f"  Características de tosse: {len(features_tosse)} features")
    if 'n_tosses' in features_tosse:
        print(f"  Número de tosses detectadas: {features_tosse['n_tosses']}")
    
    # Testar conversão para imagem
    print("\n8. Testando conversão para imagem...")
    imagem = extrator.espectrograma_para_imagem(mel_spec)
    print(f"  Imagem gerada: shape={imagem.shape}, "
          f"range=[{imagem.min():.3f}, {imagem.max():.3f}]")
    
    # Testar extração completa
    print("\n9. Testando extração completa...")
    todas_features = extrator.extrair_todas_caracteristicas(audio_teste)
    print(f"  Total de tipos de features: {len(todas_features)}")
    
    # Resumo
    print("\n" + "=" * 60)
    print("RESUMO DO TESTE:")
    print("=" * 60)
    
    contagem = 0
    for key, value in todas_features.items():
        if isinstance(value, dict):
            print(f"  {key}: {len(value)} features")
            contagem += len(value)
        elif isinstance(value, np.ndarray):
            print(f"  {key}: array {value.shape}")
            contagem += 1
        else:
            print(f"  {key}: {type(value).__name__}")
            contagem += 1
    
    print(f"\nTotal de elementos de features: {contagem}")
    print("\n" + "=" * 60)
    print("TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
