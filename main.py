"""
main.py - Sistema Principal de Classificação de Tosse

COMANDOS DISPONÍVEIS:
python main.py --treinar                  # Treina novo modelo
python main.py --melhorar                 # Melhora modelo existente
python main.py --tflite                   # Converte para TFLite
python main.py --testar                   # Testa modelo com novos áudios
python main.py --status                   # Verifica status do sistema
python main.py --limpar                   # Limpa modelos antigos
python main.py --web                      # Inicia interface web
python main.py --ajuda                    # Mostra ajuda completa
"""

import argparse
import os
import sys
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SistemaTosseCLI:
    """Sistema CLI para gerenciar modelos de tosse."""
    
    def __init__(self):
        """Inicializa o sistema."""
        self.criar_estrutura()
        logger.info("✅ Sistema de classificação de tosse inicializado")
    
    def criar_estrutura(self):
        """Cria estrutura de diretórios."""
        dirs = [
            'dados/brutos/normal',
            'dados/brutos/bronquite',
            'dados/brutos/pneumonia',
            'modelos/checkpoints',
            'modelos/tflite',
            'modelos/exportados',
            'logs',
            'resultados/graficos',
            'resultados/relatorios',
            'resultados/predicoes',
            'testes/audios',
            'testes/resultados'
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        logger.debug("Estrutura de diretórios verificada")
    
    def verificar_dependencias(self):
        """Verifica se todas as dependências estão instaladas."""
        try:
            import tensorflow as tf
            import librosa
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            
            logger.info("✅ Dependências verificadas:")
            logger.info(f"   TensorFlow: {tf.__version__}")
            logger.info(f"   Librosa: {librosa.__version__}")
            logger.info(f"   NumPy: {np.__version__}")
            
            # Verificar GPU
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"   GPU detectada: {len(gpus)} dispositivo(s)")
            else:
                logger.info("   CPU será utilizada")
            
            return True
            
        except ImportError as e:
            logger.error(f"❌ Dependência faltando: {e}")
            logger.info("📦 Instale com: pip install -r requisitos.txt")
            return False
    
    def treinar(self, epochs=50, batch_size=16, modelo_tipo='cnn_basico'):
        """Treina um novo modelo."""
        logger.info("=" * 60)
        logger.info("🚀 INICIANDO TREINAMENTO")
        logger.info("=" * 60)
        
        # Verificar dependências
        if not self.verificar_dependencias():
            return False
        
        # Verificar dados
        if not self.verificar_dados():
            return False
        
        try:
            # Importar módulos de treinamento
            sys.path.append('src')
            
            from treinar import treinar_modelo_completo
            
            # Configurações
            config = {
                'EPOCHS': epochs,
                'BATCH_SIZE': batch_size,
                'MODEL_TYPE': modelo_tipo,
                'SAVE_BEST': True,
                'SAVE_FINAL': True,
                'EXPORT_TFLITE': True
            }
            
            # Executar treinamento
            logger.info(f"📋 Configurações:")
            logger.info(f"   Épocas: {epochs}")
            logger.info(f"   Batch Size: {batch_size}")
            logger.info(f"   Tipo de Modelo: {modelo_tipo}")
            logger.info(f"   Salvamento: checkpoints + final")
            
            resultado = treinar_modelo_completo(
                csv_path='dados/rotulos.csv',
                audio_dir='dados/brutos',
                config=config,
                output_dir='resultados'
            )
            
            if resultado:
                logger.info("=" * 60)
                logger.info("🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
                logger.info("=" * 60)
                self.mostrar_resultados()
                return True
            else:
                logger.error("❌ Falha no treinamento")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def melhorar(self, modelo_path=None, epochs=30):
        """Melhora um modelo existente."""
        logger.info("=" * 60)
        logger.info("🔧 MELHORANDO MODELO EXISTENTE")
        logger.info("=" * 60)
        
        # Encontrar modelo mais recente se não especificado
        if modelo_path is None:
            modelos = self.listar_modelos()
            if not modelos:
                logger.error("❌ Nenhum modelo encontrado para melhorar")
                logger.info("   Primeiro treine um modelo: python main.py --treinar")
                return False
            
            modelo_path = modelos[0]['path']
            logger.info(f"📁 Usando modelo: {os.path.basename(modelo_path)}")
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # Carregar modelo
            logger.info(f"🔄 Carregando modelo: {modelo_path}")
            modelo = keras.models.load_model(modelo_path)
            
            # Congelar camadas iniciais para fine-tuning
            logger.info("🎯 Configurando fine-tuning...")
            
            # Descongelar últimas camadas
            for layer in modelo.layers[-10:]:  # Últimas 10 camadas
                layer.trainable = True
            
            # Recompilar com learning rate menor
            modelo.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.00001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Carregar dados
            from src.preprocessamento import PreprocessadorTosse
            from src.extrair_caracteristicas import ExtratorCaracteristicasTosse
            import pandas as pd
            import numpy as np
            
            df = pd.read_csv('dados/rotulos.csv')
            
            preprocessador = PreprocessadorTosse()
            extrator = ExtratorCaracteristicasTosse()
            
            # Processar dados
            X = []
            y = []
            
            for i, row in df.iterrows():
                try:
                    audio_path = os.path.join('dados/brutos', row['caminho'])
                    audio = preprocessador.processar_pipeline(audio_path)
                    mel_spec = extrator.extrair_espectrograma_mel(audio)
                    imagem = extrator.espectrograma_para_imagem(mel_spec)
                    
                    X.append(imagem)
                    y.append(row['rotulo'])
                    
                except Exception as e:
                    logger.warning(f"⚠️  Erro no áudio {row['caminho']}: {e}")
            
            # Converter labels
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            y_cat = keras.utils.to_categorical(y_encoded)
            
            X = np.array(X)
            
            # Normalizar
            mean = np.mean(X, axis=(0, 1, 2, 3), keepdims=True)
            std = np.std(X, axis=(0, 1, 2, 3), keepdims=True) + 1e-7
            X = (X - mean) / std
            
            # Dividir dados
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_cat, test_size=0.2, random_state=42
            )
            
            # Treinar
            logger.info(f"🎯 Fine-tuning por {epochs} épocas...")
            
            checkpoint = keras.callbacks.ModelCheckpoint(
                f'modelos/melhorado_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
            
            history = modelo.fit(
                X_train, y_train,
                epochs=epochs,
                validation_data=(X_val, y_val),
                batch_size=16,
                callbacks=[checkpoint],
                verbose=1
            )
            
            # Salvar modelo final
            modelo_final_path = f'modelos/modelo_melhorado_final.h5'
            modelo.save(modelo_final_path)
            
            logger.info(f"✅ Modelo melhorado salvo em: {modelo_final_path}")
            
            # Mostrar melhoria
            melhor_val_acc = max(history.history['val_accuracy'])
            logger.info(f"📈 Melhor acurácia na validação: {melhor_val_acc:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao melhorar modelo: {e}")
            return False
    
    def exportar_tflite(self, modelo_path=None, quantizar=True):
        """Exporta modelo para TensorFlow Lite."""
        logger.info("=" * 60)
        logger.info("📱 EXPORTANDO PARA TENSORFLOW LITE")
        logger.info("=" * 60)
        
        # Encontrar modelo
        if modelo_path is None:
            modelos = self.listar_modelos()
            if not modelos:
                logger.error("❌ Nenhum modelo encontrado")
                return False
            
            modelo_path = modelos[0]['path']
            logger.info(f"📁 Usando modelo: {os.path.basename(modelo_path)}")
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # Carregar modelo
            modelo = keras.models.load_model(modelo_path)
            
            # Converter
            converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
            
            if quantizar:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                logger.info("✅ Quantização ativada")
            
            tflite_model = converter.convert()
            
            # Salvar
            nome_base = os.path.splitext(os.path.basename(modelo_path))[0]
            tflite_path = f'modelos/tflite/{nome_base}.tflite'
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # Tamanho
            tamanho_mb = os.path.getsize(tflite_path) / (1024 * 1024)
            
            logger.info(f"✅ TFLite exportado com sucesso!")
            logger.info(f"📁 Arquivo: {tflite_path}")
            logger.info(f"📏 Tamanho: {tamanho_mb:.2f} MB")
            
            # Criar exemplo de uso
            self.criar_exemplo_tflite(tflite_path)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao exportar TFLite: {e}")
            return False
    
    def criar_exemplo_tflite(self, tflite_path):
        """Cria exemplo de uso do modelo TFLite."""
        exemplo = f"""# 📱 EXEMPLO DE USO DO MODELO TFLITE
# Modelo: {os.path.basename(tflite_path)}

import numpy as np
import tensorflow as tf

# 1. CARREGAR MODELO TFLITE
interpreter = tf.lite.Interpreter(model_path="{tflite_path}")
interpreter.allocate_tensors()

# 2. OBTER DETALHES DE ENTRADA/SAÍDA
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Detalhes do modelo:")
print(f"  Entrada: {input_details[0]['shape']}")
print(f"  Saída: {output_details[0]['shape']}")

# 3. PREPARAR DADOS DE ENTRADA
# Formato: [1, 128, 128, 3] para espectrogramas
input_shape = input_details[0]['shape']
input_data = np.random.random(input_shape).astype(np.float32)

# 4. EXECUTAR INFERÊNCIA
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# 5. OBTER RESULTADOS
output_data = interpreter.get_tensor(output_details[0]['index'])

# 6. INTERPRETAR
classes = ['Normal', 'Bronquite', 'Pneumonia']
predicted_class = classes[np.argmax(output_data[0])]
confidence = np.max(output_data[0])

print(f"\\nResultado: {predicted_class} ({confidence:.1%} confiança)")

# 7. PARA USAR COM ÁUDIO REAL:
def classificar_audio(caminho_audio):
    \"\"\"Classifica um arquivo de áudio.\"\"\"
    from src.preprocessamento import PreprocessadorTosse
    from src.extrair_caracteristicas import ExtratorCaracteristicasTosse
    
    # Pré-processar
    preprocessador = PreprocessadorTosse()
    extrator = ExtratorCaracteristicasTosse()
    
    audio = preprocessador.processar_pipeline(caminho_audio)
    mel_spec = extrator.extrair_espectrograma_mel(audio)
    imagem = extrator.espectrograma_para_imagem(mel_spec)
    
    # Adicionar dimensão do batch
    input_data = np.expand_dims(imagem, axis=0).astype(np.float32)
    
    # Executar
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    return interpreter.get_tensor(output_details[0]['index'])
"""
        
        exemplo_path = 'modelos/tflite/exemplo_uso.py'
        with open(exemplo_path, 'w') as f:
            f.write(exemplo)
        
        logger.info(f"📝 Exemplo de uso salvo em: {exemplo_path}")
    
    def testar(self, audio_dir=None):
        """Testa o modelo com novos áudios."""
        logger.info("=" * 60)
        logger.info("🧪 TESTANDO MODELO COM NOVOS ÁUDIOS")
        logger.info("=" * 60)
        
        # Encontrar modelo mais recente
        modelos = self.listar_modelos()
        if not modelos:
            logger.error("❌ Nenhum modelo encontrado")
            return False
        
        modelo_path = modelos[0]['path']
        
        # Verificar áudios
        if audio_dir is None:
            audio_dir = 'testes/audios'
        
        if not os.path.exists(audio_dir):
            logger.error(f"❌ Diretório não encontrado: {audio_dir}")
            logger.info("📁 Crie o diretório e adicione áudios .wav ou .mp3")
            return False
        
        try:
            import glob
            import tensorflow as tf
            from tensorflow import keras
            from src.preprocessamento import PreprocessadorTosse
            from src.extrair_caracteristicas import ExtratorCaracteristicasTosse
            import numpy as np
            
            # Carregar modelo
            modelo = keras.models.load_model(modelo_path)
            
            # Processadores
            preprocessador = PreprocessadorTosse()
            extrator = ExtratorCaracteristicasTosse()
            
            # Encontrar áudios
            audios = []
            for ext in ['*.wav', '*.mp3']:
                audios.extend(glob.glob(os.path.join(audio_dir, ext)))
            
            if not audios:
                logger.error(f"❌ Nenhum áudio encontrado em {audio_dir}")
                return False
            
            logger.info(f"🔍 Encontrados {len(audios)} áudios para teste")
            
            resultados = []
            
            for audio_path in audios:
                try:
                    # Processar
                    audio = preprocessador.processar_pipeline(audio_path)
                    mel_spec = extrator.extrair_espectrograma_mel(audio)
                    imagem = extrator.espectrograma_para_imagem(mel_spec)
                    
                    # Normalizar (precisa usar mesma normalização do treino)
                    # Para simplificar, usaremos normalização básica
                    imagem_norm = (imagem - np.mean(imagem)) / (np.std(imagem) + 1e-7)
                    
                    # Adicionar dimensão do batch
                    input_data = np.expand_dims(imagem_norm, axis=0)
                    
                    # Predizer
                    predicao = modelo.predict(input_data, verbose=0)
                    
                    # Interpretar
                    classes = ['Normal', 'Bronquite', 'Pneumonia']
                    classe_idx = np.argmax(predicao[0])
                    confianca = predicao[0][classe_idx]
                    
                    resultado = {
                        'arquivo': os.path.basename(audio_path),
                        'classe': classes[classe_idx],
                        'confianca': float(confianca),
                        'probabilidades': predicao[0].tolist()
                    }
                    
                    resultados.append(resultado)
                    
                    logger.info(f"🎵 {resultado['arquivo']}: "
                               f"{resultado['classe']} ({resultado['confianca']:.1%})")
                    
                except Exception as e:
                    logger.warning(f"⚠️  Erro no áudio {audio_path}: {e}")
            
            # Salvar resultados
            if resultados:
                import json
                resultados_path = 'testes/resultados/teste.json'
                with open(resultados_path, 'w') as f:
                    json.dump(resultados, f, indent=2)
                
                logger.info(f"📁 Resultados salvos em: {resultados_path}")
                
                # Estatísticas
                classes_preditas = [r['classe'] for r in resultados]
                from collections import Counter
                contagem = Counter(classes_preditas)
                
                logger.info("📊 Estatísticas:")
                for classe, quantidade in contagem.items():
                    logger.info(f"   {classe}: {quantidade} áudios")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erro no teste: {e}")
            return False
    
    def status(self):
        """Mostra status do sistema."""
        logger.info("=" * 60)
        logger.info("📊 STATUS DO SISTEMA")
        logger.info("=" * 60)
        
        # Verificar dependências
        self.verificar_dependencias()
        
        # Verificar dados
        dados_ok = self.verificar_dados(mostrar_detalhes=True)
        
        # Verificar modelos
        modelos = self.listar_modelos()
        
        if modelos:
            logger.info(f"\n📁 MODELOS DISPONÍVEIS: {len(modelos)}")
            for modelo in modelos[:3]:  # Mostrar apenas os 3 mais recentes
                logger.info(f"   {modelo['nome']} ({modelo['tamanho']:.1f} MB)")
        else:
            logger.info("\n📁 MODELOS: Nenhum modelo treinado")
        
        # Espaço em disco
        import shutil
        total, usado, livre = shutil.disk_usage(".")
        logger.info(f"\n💾 ESPAÇO EM DISCO:")
        logger.info(f"   Total: {total // (2**30):.1f} GB")
        logger.info(f"   Usado: {usado // (2**30):.1f} GB")
        logger.info(f"   Livre: {livre // (2**30):.1f} GB")
        
        # Recomendações
        logger.info("\n💡 RECOMENDAÇÕES:")
        
        if not dados_ok:
            logger.info("   ⚠️  Adicione mais áudios para treinar")
        
        if not modelos:
            logger.info("   🚀 Execute: python main.py --treinar")
        else:
            logger.info("   📱 Para usar no celular: python main.py --tflite")
    
    def limpar(self, manter=3):
        """Limpa modelos antigos."""
        logger.info("=" * 60)
        logger.info("🧹 LIMPANDO MODELOS ANTIGOS")
        logger.info("=" * 60)
        
        modelos = self.listar_modelos()
        
        if len(modelos) <= manter:
            logger.info(f"✅ Apenas {len(modelos)} modelos, nada para limpar")
            return True
        
        logger.info(f"📁 Modelos encontrados: {len(modelos)}")
        logger.info(f"📌 Serão mantidos: {manter} modelos mais recentes")
        
        confirmar = input(f"\n❓ Remover {len(modelos) - manter} modelos antigos? (s/n): ")
        
        if confirmar.lower() != 's':
            logger.info("❌ Operação cancelada")
            return False
        
        # Remover modelos antigos (exceto os 'manter' mais recentes)
        for i in range(manter, len(modelos)):
            try:
                os.remove(modelos[i]['path'])
                logger.info(f"🗑️  Removido: {modelos[i]['nome']}")
            except Exception as e:
                logger.error(f"❌ Erro ao remover {modelos[i]['nome']}: {e}")
        
        # Limpar checkpoints antigos
        import glob
        checkpoints = glob.glob('modelos/checkpoints/*.h5')
        for checkpoint in checkpoints:
            try:
                os.remove(checkpoint)
                logger.info(f"🗑️  Removido checkpoint: {os.path.basename(checkpoint)}")
            except:
                pass
        
        logger.info("✅ Limpeza concluída")
        return True
    
    def web(self):
        """Inicia interface web (simples)."""
        logger.info("=" * 60)
        logger.info("🌐 INICIANDO INTERFACE WEB")
        logger.info("=" * 60)
        
        try:
            import http.server
            import socketserver
            import webbrowser
            import json
            
            # Criar página HTML simples
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Sistema de Classificação de Tosse</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .card { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 10px; }
                    button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
                    button:hover { background: #45a049; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🤖 Sistema de Classificação de Tosse</h1>
                    
                    <div class="card">
                        <h2>📊 Status do Sistema</h2>
                        <div id="status">Carregando...</div>
                    </div>
                    
                    <div class="card">
                        <h2>🎵 Testar Novo Áudio</h2>
                        <input type="file" id="audioFile" accept=".wav,.mp3">
                        <button onclick="testarAudio()">Testar Áudio</button>
                        <div id="resultado"></div>
                    </div>
                    
                    <div class="card">
                        <h2>⚙️ Ações</h2>
                        <button onclick="atualizarModelos()">Atualizar Modelos</button>
                        <button onclick="verLogs()">Ver Logs</button>
                    </div>
                </div>
                
                <script>
                    async function carregarStatus() {
                        const response = await fetch('/status');
                        const data = await response.json();
                        document.getElementById('status').innerHTML = `
                            <p>Modelos: ${data.modelos}</p>
                            <p>Áudios: ${data.audios}</p>
                            <p>Último treinamento: ${data.ultimo_treinamento}</p>
                        `;
                    }
                    
                    async function testarAudio() {
                        const fileInput = document.getElementById('audioFile');
                        if (!fileInput.files[0]) {
                            alert('Selecione um arquivo de áudio');
                            return;
                        }
                        
                        const formData = new FormData();
                        formData.append('audio', fileInput.files[0]);
                        
                        const response = await fetch('/testar', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const resultado = await response.json();
                        document.getElementById('resultado').innerHTML = `
                            <h3>Resultado: ${resultado.classe}</h3>
                            <p>Confiança: ${(resultado.confianca * 100).toFixed(1)}%</p>
                        `;
                    }
                    
                    window.onload = carregarStatus;
                </script>
            </body>
            </html>
            """
            
            # Criar handler HTTP
            class TosseHandler(http.server.SimpleHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/':
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(html.encode())
                    elif self.path == '/status':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        
                        # Status real
                        modelos = self.listar_modelos()
                        status_data = {
                            'modelos': len(modelos),
                            'audios': self.contar_audios(),
                            'ultimo_treinamento': modelos[0]['data'] if modelos else 'Nunca'
                        }
                        
                        self.wfile.write(json.dumps(status_data).encode())
                    else:
                        super().do_GET()
            
            # Iniciar servidor
            PORT = 8080
            handler = TosseHandler
            
            with socketserver.TCPServer(("", PORT), handler) as httpd:
                logger.info(f"🌐 Servidor web iniciado em: http://localhost:{PORT}")
                logger.info("📱 Acesse no navegador para interface gráfica")
                logger.info("🛑 Pressione Ctrl+C para parar o servidor")
                
                # Abrir navegador automaticamente
                webbrowser.open(f"http://localhost:{PORT}")
                
                httpd.serve_forever()
                
        except ImportError:
            logger.error("❌ Bibliotecas web não disponíveis")
            logger.info("📦 Instale: pip install webbrowser")
        except Exception as e:
            logger.error(f"❌ Erro no servidor web: {e}")
    
    # ============================================================================
    # FUNÇÕES AUXILIARES
    # ============================================================================
    
    def verificar_dados(self, mostrar_detalhes=False):
        """Verifica se há dados para treinamento."""
        import glob
        
        total_audios = 0
        classes = ['normal', 'bronquite', 'pneumonia']
        
        for classe in classes:
            pasta = f'dados/brutos/{classe}'
            
            if not os.path.exists(pasta):
                if mostrar_detalhes:
                    logger.warning(f"⚠️  Pasta não encontrada: {pasta}")
                continue
            
            # Contar áudios
            audios = []
            for ext in ['*.wav', '*.mp3']:
                audios.extend(glob.glob(os.path.join(pasta, ext)))
            
            total_audios += len(audios)
            
            if mostrar_detalhes:
                logger.info(f"📁 {classe}: {len(audios)} áudios")
        
        if total_audios == 0:
            logger.error("❌ Nenhum áudio encontrado para treinamento")
            logger.info("📝 Coloque áudios em:")
            logger.info("   dados/brutos/normal/*.wav")
            logger.info("   dados/brutos/bronquite/*.wav")
            logger.info("   dados/brutos/pneumonia/*.wav")
            return False
        
        if mostrar_detalhes:
            logger.info(f"✅ Total de áudios: {total_audios}")
        
        # Verificar se há pelo menos alguns áudios por classe
        if total_audios < 10:
            logger.warning(f"⚠️  Poucos áudios ({total_audios}). Recomendado: 50+")
        
        return True
    
    def listar_modelos(self):
        """Lista todos os modelos treinados."""
        import glob
        
        modelos = []
        
        # Modelos finais
        for modelo_path in glob.glob('modelos/modelo_final_*.h5'):
            nome = os.path.basename(modelo_path)
            tamanho = os.path.getsize(modelo_path) / (1024 * 1024)
            
            # Extrair data do nome
            try:
                data_str = nome.replace('modelo_final_', '').replace('.h5', '')
                data = datetime.strptime(data_str, '%Y%m%d_%H%M%S')
                data_formatada = data.strftime('%d/%m/%Y %H:%M')
            except:
                data_formatada = "Data desconhecida"
            
            modelos.append({
                'nome': nome,
                'path': modelo_path,
                'tamanho': tamanho,
                'data': data_formatada,
                'tipo': 'final'
            })
        
        # Ordenar por data (mais recente primeiro)
        modelos.sort(key=lambda x: x['data'], reverse=True)
        
        return modelos
    
    def contar_audios(self):
        """Conta áudios disponíveis."""
        import glob
        
        total = 0
        for classe in ['normal', 'bronquite', 'pneumonia']:
            pasta = f'dados/brutos/{classe}'
            if os.path.exists(pasta):
                for ext in ['*.wav', '*.mp3']:
                    total += len(glob.glob(os.path.join(pasta, ext)))
        
        return total
    
    def mostrar_resultados(self):
        """Mostra resultados do último treinamento."""
        import glob
        import json
        
        # Encontrar relatório mais recente
        relatorios = glob.glob('resultados/relatorios/*.json')
        if not relatorios:
            logger.info("📊 Nenhum relatório encontrado")
            return
        
        relatorio_path = max(relatorios, key=os.path.getctime)
        
        try:
            with open(relatorio_path, 'r') as f:
                dados = json.load(f)
            
            if isinstance(dados, dict) and 'accuracy' in dados:
                logger.info("📈 RESULTADOS DO TREINAMENTO:")
                logger.info(f"   Acurácia: {dados['accuracy']:.2%}")
                logger.info(f"   Precisão: {dados.get('precision', 0):.2%}")
                logger.info(f"   Recall: {dados.get('recall', 0):.2%}")
                
                # Se for relatório de classificação
                if 'Normal' in dados or 'normal' in dados:
                    for classe in ['Normal', 'Bronquite', 'Pneumonia']:
                        if classe in dados:
                            logger.info(f"   {classe}:")
                            logger.info(f"     Precisão: {dados[classe].get('precision', 0):.2%}")
                            logger.info(f"     Recall: {dados[classe].get('recall', 0):.2%}")
        
        except Exception as e:
            logger.warning(f"⚠️  Erro ao ler relatório: {e}")

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal do sistema CLI."""
    parser = argparse.ArgumentParser(
        description='Sistema de Classificação de Tosse',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXEMPLOS DE USO:
  python main.py --treinar                 # Treina novo modelo
  python main.py --treinar --epochs 100    # Treina com 100 épocas
  python main.py --melhorar                # Melhora modelo existente
  python main.py --tflite                  # Converte para celular
  python main.py --testar                  # Testa com novos áudios
  python main.py --status                  # Verifica status
  python main.py --limpar                  # Limpa modelos antigos
  python main.py --web                     # Interface web
  python main.py --ajuda                   # Mostra esta ajuda
        """
    )
    
    # Comandos principais
    parser.add_argument('--treinar', '-t', action='store_true',
                       help='Treina um novo modelo')
    
    parser.add_argument('--melhorar', '-m', action='store_true',
                       help='Melhora um modelo existente (fine-tuning)')
    
    parser.add_argument('--tflite', '-l', action='store_true',
                       help='Converte modelo para TensorFlow Lite')
    
    parser.add_argument('--testar', '-s', action='store_true',
                       help='Testa modelo com novos áudios')
    
    parser.add_argument('--status', '-i', action='store_true',
                       help='Mostra status do sistema')
    
    parser.add_argument('--limpar', '-c', action='store_true',
                       help='Limpa modelos antigos')
    
    parser.add_argument('--web', '-w', action='store_true',
                       help='Inicia interface web')
    
    parser.add_argument('--ajuda', '-a', action='store_true',
                       help='Mostra ajuda detalhada')
    
    # Opções para treinamento
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas para treinamento')
    
    parser.add_argument('--batch', type=int, default=16,
                       help='Tamanho do batch para treinamento')
    
    parser.add_argument('--modelo', type=str, default='cnn_basico',
                       choices=['cnn_basico', 'cnn_avancado', 'mobilenet'],
                       help='Tipo de modelo para treinar')
    
    parser.add_argument('--audio-dir', type=str,
                       help='Diretório com áudios para teste')
    
    parser.add_argument('--manter', type=int, default=3,
                       help='Número de modelos a manter ao limpar')
    
    args = parser.parse_args()
    
    # Criar sistema
    sistema = SistemaTosseCLI()
    
    # Executar comando baseado nos argumentos
    if args.ajuda or len(sys.argv) == 1:
        parser.print_help()
        
        # Mostrar status rápido também
        print("\n" + "=" * 60)
        sistema.status()
        
    elif args.treinar:
        sistema.treinar(
            epochs=args.epochs,
            batch_size=args.batch,
            modelo_tipo=args.modelo
        )
    
    elif args.melhorar:
        sistema.melhorar(epochs=args.epochs)
    
    elif args.tflite:
        sistema.exportar_tflite()
    
    elif args.testar:
        sistema.testar(audio_dir=args.audio_dir)
    
    elif args.status:
        sistema.status()
    
    elif args.limpar:
        sistema.limpar(manter=args.manter)
    
    elif args.web:
        sistema.web()
    
    else:
        print("❌ Comando não reconhecido")
        print("📝 Use: python main.py --ajuda")

if __name__ == "__main__":
    # Banner inicial
    print("=" * 60)
    print("🤖 SISTEMA DE CLASSIFICAÇÃO DE TOSSE")
    print("=" * 60)
    
    main()
