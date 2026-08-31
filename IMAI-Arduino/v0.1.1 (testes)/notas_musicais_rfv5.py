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


# CONFIGURAÇÕES GERAIS

PORTA_SERIAL   = 'COM9'
ARQUIVO_MODELO = r"C:\Users\INFORMÁTICA\Documents\projeto ruan 3 bimestre\IMAI-Arduino\v0.1.1 (testes)\modelo_notas.pkl"
# ARQUIVO_MODELO = "modelo_notas.pkl"

MAPA_NOTA_COMANDO = {
    "Do (C)": "C", "Re (D)": "D", "Mi (E)": "E", "Fa (F)": "F",
    "Sol (G)": "G", "La (A)": "A", "Si (B)": "B", "Silencio": "X"
}

# --- CONFIGURAÇÃO DO METRÔNOMO ---
BPM = 120  # BPM inicial padrão
FIGURAS = {
    ord('1'): ("Seminima",    1.0),   # 1 batida
    ord('2'): ("Minima",      2.0),   # 2 batidas
    ord('3'): ("Semibreve",   4.0),   # 4 batidas
    ord('4'): ("Colcheia",    0.5),   # meia batida
    ord('5'): ("Semicolcheia", 0.25)  # quarto de batida
}

figura_atual = "Seminima"
multiplicador_figura = 1.0
intervalo_batida = (60.0 / BPM) * multiplicador_figura
ultimo_tempo_batida = time.time()

# CONEXÃO SERIAL COM ARDUINO / ESP32
try:
    arduino = serial.Serial(PORTA_SERIAL, 9600, timeout=0.1)
    time.sleep(2)
    print("Arduino/ESP32 conectado com sucesso!")
except Exception as e:
    print(f"Arduino/ESP32 não encontrado: {e}. Rodando em modo visual.")
    arduino = None


# CARREGAMENTO DO MODELO DE IA

if not os.path.exists(ARQUIVO_MODELO):
    print(f"Modelo '{ARQUIVO_MODELO}' não encontrado. Execute o treino primeiro.")
    exit()

with open(ARQUIVO_MODELO, "rb") as f:
    clf = pickle.load(f)
print("Modelo de IA carregado.")

# SETUP DO MEDIAPIPE (Com suporte a pastas com acento no Windows)

caminho_atual = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(caminho_atual, '..', 'hand_landmarker.task')

if not os.path.exists(modelo_path):
    print(f"Arquivo '{modelo_path}' não encontrado!")
    exit()

with open(modelo_path, "rb") as f:
    modelo_bytes = f.read()

base_options = python.BaseOptions(model_asset_buffer=modelo_bytes)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(buscar_indice_camera_por_nome(NOME_DA_CAMERA_DESEJADA), cv2.CAP_DSHOW)


def extrair_features(pontos_mao):
    indices_palma = [0, 5, 9, 13, 17]
    cx = sum(pontos_mao[i].x for i in indices_palma) / len(indices_palma)
    cy = sum(pontos_mao[i].y for i in indices_palma) / len(indices_palma)
    cz = sum(pontos_mao[i].z for i in indices_palma) / len(indices_palma)

    p0 = pontos_mao[0]
    p9 = pontos_mao[9]
    dist_referencia = np.sqrt((p0.x - p9.x)**2 + (p0.y - p9.y)**2 + (p0.z - p9.z)**2)

    if dist_referencia == 0:
        dist_referencia = 1.0

    features = []
    for p in pontos_mao:
        features.append((p.x - cx) / dist_referencia)
        features.append((p.y - cy) / dist_referencia)
        features.append((p.z - cz) / dist_referencia)

    return np.array(features)

# GERENCIAMENTO DE THREADS

frame_lock = threading.Lock()
resultado_lock = threading.Lock()
novo_frame_evento = threading.Event()
parar_evento = threading.Event()

frame_para_processar = None
nota_detectada = "Silencio"
confianca = 0.0
pontos_mao_para_desenho = None


def thread_mediapipe():
    global nota_detectada, confianca, pontos_mao_para_desenho, frame_para_processar

    while not parar_evento.is_set():
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

# =========================================================================
# LOOP PRINCIPAL
# =========================================================================
while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    with frame_lock:
        frame_para_processar = frame.copy()
    novo_frame_evento.set()

    with resultado_lock:
        nota_local = nota_detectada
        confianca_local = confianca
        pontos_local = pontos_mao_para_desenho

    if pontos_local:
        for pontos_mao in pontos_local:
            for ponto in pontos_mao:
                px, py = int(ponto.x * w), int(ponto.y * h)
                cv2.circle(frame, (px, py), 5, (229, 136, 30), -1)

    # --- Painel de HUD ---
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

    cv2.putText(frame, f"Figura: {figura_atual}", (w - 180, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 92, 0), 1)

    # --- METRÔNOMO ---
    tempo_atual = time.time()

    if tempo_atual - ultimo_tempo_batida >= intervalo_batida:
        ultimo_tempo_batida = tempo_atual

        if arduino and nota_local != "Silencio":
            comando = MAPA_NOTA_COMANDO.get(nota_local, "X") + "\n"
            try:
                arduino.write(comando.encode())
            except Exception as e:
                print(f"Erro serial: {e}")
                arduino = None

    # Indicador do Metrônomo
    est_no_clique = (tempo_atual - ultimo_tempo_batida) < 0.10
    cor_metronomo = (0, 255, 0) if est_no_clique else (50, 50, 50)

    cv2.circle(frame, (w - 40, 40), 15, cor_metronomo, -1)
    cv2.putText(frame, f"BPM: {BPM}", (w - 120, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

    cv2.imshow("Libras Musical AI", frame)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord('q'):
        break

    # Teclas + e = aumentam o BPM
    if tecla in (ord('+'), ord('=')):
        BPM = min(BPM + 5, 300)
        intervalo_batida = (60.0 / BPM) * multiplicador_figura

    # Tecla - diminui o BPM
    if tecla == ord('-'):
        BPM = max(BPM - 5, 30)
        intervalo_batida = (60.0 / BPM) * multiplicador_figura

    # Teclas 1 a 5 mudam a figura rítmica
    if tecla in FIGURAS:
        figura_atual, multiplicador_figura = FIGURAS[tecla]
        intervalo_batida = (60.0 / BPM) * multiplicador_figura
        print(f"Figura: {figura_atual} | Intervalo: {intervalo_batida:.2f}s")

# --- ENCERRAMENTO ---
parar_evento.set()
novo_frame_evento.set()
worker.join(timeout=2)

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
detector.close()