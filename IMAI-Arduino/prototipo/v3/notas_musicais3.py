import cv2
import mediapipe as mp
import serial
import time
import math
import numpy as np
import os
import json
import pickle
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pygrabber.dshow_graph import FilterGraph
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =========================================================================
# CONFIGURAÇÃO
# =========================================================================
NOME_DA_CAMERA_DESEJADA = "OBS Virtual Camera"
PORTA_SERIAL = 'COM13'
ARQUIVO_DADOS = "dados_landmarks.json"
ARQUIVO_MODELO = "modelo_notas.pkl"

NOTAS = ["Do (C)", "Re (D)", "Mi (E)", "Fa (F)", "Sol (G)", "La (A)", "Si (B)", "Silencio"]
MAPA_NOTA_COMANDO = {
    "Do (C)": "C", "Re (D)": "D", "Mi (E)": "E", "Fa (F)": "F",
    "Sol (G)": "G", "La (A)": "A", "Si (B)": "B", "Silencio": "X"
}

# =========================================================================
# MODO: 'coletar' para gravar amostras | 'treinar' para treinar | 'tocar' para usar
# =========================================================================
MODO = "tocar"
NOTA_SENDO_COLETADA = "Do (C)"   # Altere ao coletar cada nota
# =========================================================================


def buscar_indice_camera_por_nome(nome_procurado):
    graph = FilterGraph()
    dispositivos = graph.get_input_devices()
    for idx, nome in enumerate(dispositivos):
        if nome_procurado.lower() in nome.lower():
            return idx
    return 0


def extrair_features(pontos_mao):
    """
    Extrai 63 features (21 pontos × x,y,z) normalizadas pelo centro da palma.
    Centro = média dos landmarks p[0], p[5], p[9], p[13], p[17].
    """
    indices_palma = [0, 5, 9, 13, 17]
    cx = sum(pontos_mao[i].x for i in indices_palma) / len(indices_palma)
    cy = sum(pontos_mao[i].y for i in indices_palma) / len(indices_palma)
    cz = sum(pontos_mao[i].z for i in indices_palma) / len(indices_palma)

    features = []
    for p in pontos_mao:
        features.append(p.x - cx)
        features.append(p.y - cy)
        features.append(p.z - cz)
    return np.array(features)


def carregar_modelo():
    if os.path.exists(ARQUIVO_MODELO):
        with open(ARQUIVO_MODELO, "rb") as f:
            return pickle.load(f)
    return None


def treinar_modelo():
    if not os.path.exists(ARQUIVO_DADOS):
        print("Arquivo de dados não encontrado. Rode no MODO='coletar' primeiro.")
        return None

    with open(ARQUIVO_DADOS, "r") as f:
        dados = json.load(f)

    X = np.array([d["features"] for d in dados])
    y = [d["label"] for d in dados]

    print(f"\nTotal de amostras: {len(y)}")
    for nota in NOTAS:
        print(f"  {nota}: {y.count(nota)} amostras")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)

    print("\n--- Relatório de Classificação ---")
    print(classification_report(y_test, clf.predict(X_test)))

    with open(ARQUIVO_MODELO, "wb") as f:
        pickle.dump(clf, f)
    print(f"Modelo salvo em '{ARQUIVO_MODELO}'")
    return clf


# =========================================================================
# SETUP
# =========================================================================
try:
    arduino = serial.Serial(PORTA_SERIAL, 9600, timeout=0.1)
    time.sleep(2)
    print("Arduino conectado!")
except Exception as e:
    print(f"Erro ao conectar no Arduino: {e}. Rodando em modo visual.")
    arduino = None

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8
)
detector = vision.HandLandmarker.create_from_options(options)

if MODO == "treinar":
    treinar_modelo()
    exit()

clf = carregar_modelo() if MODO == "tocar" else None
if MODO == "tocar" and clf is None:
    print("Modelo não encontrado. Treine primeiro com MODO='treinar'.")
    exit()

dados_coletados = []
if MODO == "coletar" and os.path.exists(ARQUIVO_DADOS):
    with open(ARQUIVO_DADOS, "r") as f:
        dados_coletados = json.load(f)

indice_camera = buscar_indice_camera_por_nome(NOME_DA_CAMERA_DESEJADA)
cap = cv2.VideoCapture(indice_camera, cv2.CAP_DSHOW)

time.sleep(1.5)
for tentativa in range(5):
    sucesso, frame = cap.read()
    if sucesso:
        break
    time.sleep(0.5)

ultima_nota = ""
amostras_coletadas_sessao = 0

print(f"\nMODO ATIVO: {MODO.upper()}")
if MODO == "coletar":
    print(f"Coletando para: '{NOTA_SENDO_COLETADA}' — pressione [S] para salvar amostra, [Q] para sair")

# =========================================================================
# LOOP PRINCIPAL
# =========================================================================
while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    resultado = detector.detect(mp_image)

    nota_detectada = "Silencio"
    confianca = 0.0

    if resultado.hand_landmarks:
        for pontos_mao in resultado.hand_landmarks:
            # Desenha pontos
            for ponto in pontos_mao:
                cx, cy = int(ponto.x * w), int(ponto.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            features = extrair_features(pontos_mao)

            if MODO == "tocar" and clf is not None:
                proba = clf.predict_proba([features])[0]
                idx_max = np.argmax(proba)
                confianca = proba[idx_max]
                # Só aceita predição com confiança mínima de 60%
                if confianca >= 0.60:
                    nota_detectada = clf.classes_[idx_max]
                else:
                    nota_detectada = "Silencio"

    # HUD
    cv2.rectangle(frame, (10, 20), (400, 100), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f'NOTA: {nota_detectada}', (20, 65),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

    if MODO == "tocar":
        cv2.putText(frame, f'Confianca: {confianca:.0%}', (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if MODO == "coletar":
        cv2.putText(frame, f'Coletando: {NOTA_SENDO_COLETADA} | Amostras: {amostras_coletadas_sessao}',
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Envio para Arduino (somente no modo tocar)
    if MODO == "tocar" and arduino and nota_detectada != ultima_nota:
        comando = MAPA_NOTA_COMANDO.get(nota_detectada, "X") + "\n"
        try:
            arduino.write(comando.encode())
            ultima_nota = nota_detectada
        except Exception as e:
            print(f"Erro serial: {e}")
            arduino = None

    cv2.imshow("Libras Musical AI", frame)
    tecla = cv2.waitKey(1) & 0xFF

    # Coleta de amostras com tecla S
    if MODO == "coletar" and tecla == ord('s') and resultado.hand_landmarks:
        for pontos_mao in resultado.hand_landmarks:
            features = extrair_features(pontos_mao).tolist()
            dados_coletados.append({"label": NOTA_SENDO_COLETADA, "features": features})
            amostras_coletadas_sessao += 1
            print(f"Amostra {amostras_coletadas_sessao} salva para '{NOTA_SENDO_COLETADA}'")

    if tecla == ord('q'):
        break

# Salva dados ao sair no modo coletar
if MODO == "coletar" and dados_coletados:
    with open(ARQUIVO_DADOS, "w") as f:
        json.dump(dados_coletados, f)
    print(f"\n{amostras_coletadas_sessao} novas amostras salvas. Total no arquivo: {len(dados_coletados)}")

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
detector.close()