import cv2
import mediapipe as mp
import serial
import time
import numpy as np
import os
import pickle
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pygrabber.dshow_graph import FilterGraph

def buscar_indice_camera_por_nome(nome_procurado):
    graph = FilterGraph()
    for idx, nome in enumerate(graph.get_input_devices()):
        if nome_procurado.lower() in nome.lower():
            return idx
    return 0

NOME_DA_CAMERA_DESEJADA = "OBS Virtual Camera"
# CONFIGURAÇÃO
# =========================================================================
PORTA_SERIAL   = 'COM9'
ARQUIVO_MODELO = "modelo_notas.pkl"

MAPA_NOTA_COMANDO = {
    "Do (C)": "C", "Re (D)": "D", "Mi (E)": "E", "Fa (F)": "F",
    "Sol (G)": "G", "La (A)": "A", "Si (B)": "B", "Silencio": "X"
}

# --- CONFIGURAÇÃO DO METRÔNOMO ---
BPM = 180  # 90 batidas por minuto (1 batida a cada ~0.66 segundos)
FIGURAS = {
    ord('1'): ("Seminima",   1.0),   # 1 batida
    ord('2'): ("Minima",     2.0),   # 2 batidas
    ord('3'): ("Semibreve",  4.0),   # 4 batidas
    ord('4'): ("Colcheia",   0.5),   # meia batida
    ord('5'): ("Semicolcheia", 0.25) # quarto de batida
}

figura_atual = "Seminima"
multiplicador_figura = 1.0
intervalo_batida = (60 / BPM) * multiplicador_figura
ultimo_tempo_batida = time.time()

try:
    arduino = serial.Serial(PORTA_SERIAL, 9600, timeout=0.1)
    time.sleep(2)
    print("Arduino conectado!")
except Exception as e:
    print(f"Arduino não encontrado: {e}. Rodando em modo visual.")
    arduino = None

if not os.path.exists(ARQUIVO_MODELO):
    print("Modelo não encontrado. Rode coletar_treinar.py primeiro.")
    exit()

with open(ARQUIVO_MODELO, "rb") as f:
    clf = pickle.load(f)
print("Modelo carregado.")

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(buscar_indice_camera_por_nome(NOME_DA_CAMERA_DESEJADA), cv2.CAP_DSHOW)


def extrair_features(pontos_mao):
    # 1. Mantém a sua lógica original de centralização (Invariância de Posição)
    indices_palma = [0, 5, 9, 13, 17]
    cx = sum(pontos_mao[i].x for i in indices_palma) / len(indices_palma)
    cy = sum(pontos_mao[i].y for i in indices_palma) / len(indices_palma)
    cz = sum(pontos_mao[i].z for i in indices_palma) / len(indices_palma)

    # 2. Tamanho da mão na tela (distância geométrica do Pulso até o Dedo Médio)
    p0 = pontos_mao[0]  # Pulso
    p9 = pontos_mao[9]  # Base do dedo médio
    dist_referencia = np.sqrt((p0.x - p9.x)**2 + (p0.y - p9.y)**2 + (p0.z - p9.z)**2)

    if dist_referencia == 0:
        dist_referencia = 1.0

    features = []
    for p in pontos_mao:
        features.append((p.x - cx) / dist_referencia)
        features.append((p.y - cy) / dist_referencia)
        features.append((p.z - cz) / dist_referencia)

    return np.array(features)


frame_lock = threading.Lock()
resultado_lock = threading.Lock()
novo_frame_evento = threading.Event()
parar_evento = threading.Event()

frame_para_processar = None      # último frame cru capturado (sem HUD)
nota_detectada = "Silencio"
confianca = 0.0
pontos_mao_para_desenho = None   # landmarks da última detecção, pra desenhar os pontinhos


def thread_mediapipe():
    global nota_detectada, confianca, pontos_mao_para_desenho, frame_para_processar

    while not parar_evento.is_set():
        # espera até ter frame novo, sem ficar em busy-waiting consumindo CPU à toa
        if not novo_frame_evento.wait(timeout=0.5):
            continue
        novo_frame_evento.clear()

        with frame_lock:
            frame = frame_para_processar
        if frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        resultado = detector.detect(mp_image)

        nota_local = "Silencio"
        confianca_local = 0.0
        pontos_local = None

        # mesma lógica de classificação do arquivo original, só que rodando aqui
        if resultado.hand_landmarks:
            pontos_local = resultado.hand_landmarks
            for pontos_mao in resultado.hand_landmarks:
                features = extrair_features(pontos_mao)
                proba = clf.predict_proba([features])[0]
                idx_max = np.argmax(proba)
                confianca_local = proba[idx_max]
                if confianca_local >= 0.60:
                    nota_local = clf.classes_[idx_max]

        with resultado_lock:
            nota_detectada = nota_local
            confianca = confianca_local
            pontos_mao_para_desenho = pontos_local


worker = threading.Thread(target=thread_mediapipe, daemon=True)
worker.start()
# ============================================================================

ultima_nota = ""

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # entrega o frame mais recente pra thread do MediaPipe processar
    with frame_lock:
        frame_para_processar = frame.copy()
    novo_frame_evento.set()

    # lê o último resultado já pronto (pode ser de 1-2 frames atrás, isso é esperado
    # e é justamente o que dá fluidez à exibição)
    with resultado_lock:
        nota_local = nota_detectada
        confianca_local = confianca
        pontos_local = pontos_mao_para_desenho

    if pontos_local:
        for pontos_mao in pontos_local:
            for ponto in pontos_mao:
                px, py = int(ponto.x * w), int(ponto.y * h)
                cv2.circle(frame, (px, py), 5, (229, 136, 30), -1)

    # --- Painel translúcido com borda colorida e barra de confiança ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (350, 120), (0, 0, 0), cv2.FILLED)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.rectangle(frame, (20, 20), (350, 120), (140, 89, 78), 2, cv2.LINE_AA)

    cv2.putText(frame, f'NOTA: {nota_local}', (44, 68),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    largura_barra = int(220 * confianca_local)
    cv2.rectangle(frame, (44, 86), (264, 92), (207, 148, 135), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (44, 86), (44 + largura_barra, 92), (255, 255, 255), -1, cv2.LINE_AA)

    cv2.putText(frame, f'Confianca: {confianca_local:.0%}', (44, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (207, 148, 135), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Figura: {figura_atual}", (w - 160, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 92, 0), 1)
    
    # LÓGICA DO METRÔNOMO
    tempo_atual = time.time()
    
    # A cada batida do tempo (ex: a cada 0.66 segundos no BPM 90)
    if tempo_atual - ultimo_tempo_batida >= intervalo_batida:
        ultimo_tempo_batida = tempo_atual  # Reinicia o tempo para a próxima batida
        
        # Só dispara o som se a mão estiver fazendo alguma nota
        if arduino and nota_local != "Silencio":
            comando = MAPA_NOTA_COMANDO.get(nota_local, "X") + "\n"
            try:
                arduino.write(comando.encode())
            except Exception as e:
                print(f"Erro serial: {e}")
                arduino = None
            
    # Bolinha piscando no ritmo do BPM (Pisca nos primeiros 100ms da batida)
    est_no_clique = (tempo_atual - ultimo_tempo_batida) < 0.10
    cor_metronomo = (0, 255, 0) if est_no_clique else (50, 50, 50)
    
    # Desenha um círculo no canto superior direito para representar a batida
    cv2.circle(frame, (w - 40, 40), 15, cor_metronomo, -1)
    cv2.putText(frame, f"BPM: {BPM}", (w - 110, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

    cv2.imshow("Libras Musical AI (thread teste)", frame)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord('q'):
        break

    # Seta cima — aumenta BPM
    if tecla == 82:  # cv2.waitKey retorna 82 para seta cima
        BPM = min(BPM + 5, 300)
        intervalo_batida = (60.0 / BPM) * multiplicador_figura

    # Seta baixo — diminui BPM
    if tecla == 84:  # 84 para seta baixo
        BPM = max(BPM - 5, 20)
        intervalo_batida = (60.0 / BPM) * multiplicador_figura

    # Figuras musicais — teclas 1 a 5
    if tecla in FIGURAS:
        figura_atual, multiplicador_figura = FIGURAS[tecla]
        intervalo_batida = (60.0 / BPM) * multiplicador_figura
        print(f"Figura: {figura_atual} | Intervalo: {intervalo_batida:.2f}s")

# --- Encerramento limpo ---
parar_evento.set()
novo_frame_evento.set()  # acorda o worker caso ele esteja esperando, pra ele ver o parar_evento
worker.join(timeout=2)

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
detector.close()