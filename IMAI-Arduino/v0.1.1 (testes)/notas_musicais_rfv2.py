import cv2
import mediapipe as mp
import serial
import time
import numpy as np
import os
import pickle
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
# =========================================================================

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
ultima_nota = ""


def extrair_features(pontos_mao):
    # 1. Mantém a sua lógica original de centralização (Invariância de Posição)
    indices_palma = [0, 5, 9, 13, 17]
    cx = sum(pontos_mao[i].x for i in indices_palma) / len(indices_palma)
    cy = sum(pontos_mao[i].y for i in indices_palma) / len(indices_palma)
    cz = sum(pontos_mao[i].z for i in indices_palma) / len(indices_palma)
    
    # 2. NOVA LÓGICA: Calcula o tamanho da mão na tela (distância geométrica do Pulso até o Dedo Médio)
    p0 = pontos_mao[0]  # Pulso
    p9 = pontos_mao[9]  # Base do dedo médio
    dist_referencia = np.sqrt((p0.x - p9.x)**2 + (p0.y - p9.y)**2 + (p0.z - p9.z)**2)
    
    # Segurança matemática: evita divisão por zero caso o MediaPipe falhe bizarramente
    if dist_referencia == 0: 
        dist_referencia = 1.0

    features = []
    for p in pontos_mao:
        # Agora, além de centralizar, dividimos o resultado pelo tamanho da mão (Invariância de Escala)
        features.append((p.x - cx) / dist_referencia)
        features.append((p.y - cy) / dist_referencia)
        features.append((p.z - cz) / dist_referencia)
        
    return np.array(features)


while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    resultado = detector.detect(mp_image)

    nota_detectada = "Silencio"
    confianca = 0.0

    if resultado.hand_landmarks:
        for pontos_mao in resultado.hand_landmarks:
            for ponto in pontos_mao:
                px, py = int(ponto.x * w), int(ponto.y * h)
                cv2.circle(frame, (px, py), 5, (229, 136, 30), -1)

            features = extrair_features(pontos_mao)
            proba = clf.predict_proba([features])[0]
            idx_max = np.argmax(proba)
            confianca = proba[idx_max]
            if confianca >= 0.60:
                nota_detectada = clf.classes_[idx_max]

    # --- Painel translúcido com borda colorida e barra de confiança ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (350, 120), (0, 0, 0), cv2.FILLED)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.rectangle(frame, (20, 20), (350, 120), (140, 89, 78), 2, cv2.LINE_AA)  # borda verde-teal

    cv2.putText(frame, f'NOTA: {nota_detectada}', (44, 68),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # Barra de confiança
    largura_barra = int(220 * confianca)
    cv2.rectangle(frame, (44, 86), (264, 92), (207, 148, 135), -1, cv2.LINE_AA)       # fundo da barra
    cv2.rectangle(frame, (44, 86), (44 + largura_barra, 92), (255, 255, 255), -1, cv2.LINE_AA)  # preenchimento

    cv2.putText(frame, f'Confianca: {confianca:.0%}', (44, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (207, 148, 135), 1, cv2.LINE_AA)

    if arduino and nota_detectada != ultima_nota:
        comando = MAPA_NOTA_COMANDO.get(nota_detectada, "X") + "\n"
        try:
            arduino.write(comando.encode())
            ultima_nota = nota_detectada
        except Exception as e:
            print(f"Erro serial: {e}")
            arduino = None

    cv2.imshow("Libras Musical AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
detector.close()