import cv2
import mediapipe as mp
import numpy as np
import os
import json
import pickle
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pygrabber.dshow_graph import FilterGraph

def buscar_indice_camera_por_nome(nome_procurado):
    graph = FilterGraph()
    for idx, nome in enumerate(graph.get_input_devices()):
        if nome_procurado.lower() in nome.lower():
            return idx
    return 0

NOME_DA_CAMERA_DESEJADA = "OBS Virtual Camera"

# CONFIGURAÇÃO — altere apenas isso

NOTA_SENDO_COLETADA = "Do (C)"

NOTAS          = ["Do (C)", "Re (D)", "Mi (E)", "Fa (F)", "Sol (G)", "La (A)", "Si (B)", "Silencio"]
ARQUIVO_DADOS  = "dados_landmarks.json"
ARQUIVO_MODELO = "modelo_notas.pkl"

LIMITE_AUTO    = 500   # Máximo de amostras por sessão automática — altere aqui
INTERVALO_AUTO = 0.1   # Segundos entre cada captura automática — altere aqui
# =========================================================================


def extrair_features(pontos_mao):
    # Normaliza pela média dos pontos da palma (invariante a posição e escala)
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


def treinar():
    if not os.path.exists(ARQUIVO_DADOS):
        print("Nenhum dado coletado ainda.")
        return

    with open(ARQUIVO_DADOS, "r") as f:
        dados = json.load(f)

    X = np.array([d["features"] for d in dados])
    y = [d["label"] for d in dados]

    print(f"\nTotal de amostras: {len(y)}")
    for nota in NOTAS:
        print(f"  {nota}: {y.count(nota)} amostras")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=42
    )
    clf.fit(X_train, y_train)

    print("\n--- Relatório ---")
    print(classification_report(y_test, clf.predict(X_test)))

    with open(ARQUIVO_MODELO, "wb") as f:
        pickle.dump(clf, f)
    print(f"Modelo salvo em '{ARQUIVO_MODELO}'")


# =========================================================================
# SETUP MEDIAPIPE
# =========================================================================
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(buscar_indice_camera_por_nome(NOME_DA_CAMERA_DESEJADA), cv2.CAP_DSHOW)

dados_coletados = []
if os.path.exists(ARQUIVO_DADOS):
    with open(ARQUIVO_DADOS, "r") as f:
        dados_coletados = json.load(f)

amostras_sessao  = 0
nota_idx         = NOTAS.index(NOTA_SENDO_COLETADA)
modo_auto        = False   # True = captura automática ativa
ultimo_capture   = 0.0     # Timestamp da última captura automática
amostras_auto    = 0       # Contador da sessão automática atual

print("S = salvar manual | A = iniciar/parar auto | N = próxima nota | P = nota anterior | T = treinar | Q = sair")

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    resultado = detector.detect(mp_image)

    if resultado.hand_landmarks:
        for pontos_mao in resultado.hand_landmarks:
            for ponto in pontos_mao:
                cx, cy = int(ponto.x * w), int(ponto.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    nota_atual  = NOTAS[nota_idx]
    total_nota  = sum(1 for d in dados_coletados if d["label"] == nota_atual)
    agora       = time.time()

    # --- Captura automática ---
    # Salva uma amostra a cada INTERVALO_AUTO segundos enquanto modo_auto estiver ativo
    if modo_auto and resultado.hand_landmarks:
        if agora - ultimo_capture >= INTERVALO_AUTO:
            if amostras_auto < LIMITE_AUTO:
                for pontos_mao in resultado.hand_landmarks:
                    features = extrair_features(pontos_mao).tolist()
                    dados_coletados.append({"label": nota_atual, "features": features})
                    amostras_sessao += 1
                    amostras_auto   += 1
                    ultimo_capture   = agora
                    print(f"[AUTO][{nota_atual}] {amostras_auto}/{LIMITE_AUTO}")
            else:
                # Atingiu o limite — para automaticamente e salva
                modo_auto = False
                with open(ARQUIVO_DADOS, "w") as f:
                    json.dump(dados_coletados, f)
                print(f"Limite de {LIMITE_AUTO} amostras atingido. Auto desativado e dados salvos.")

    # HUD — linha 1: nota atual e progresso
    cv2.rectangle(frame, (10, 10), (600, 110), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"Nota: {nota_atual} [{nota_idx+1}/{len(NOTAS)}]",
                (20, 48), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

    # HUD — linha 2: contadores
    cv2.putText(frame, f"Total: {total_nota} | Sessao: {amostras_sessao}",
                (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # HUD — linha 3: status do modo automático
    if modo_auto:
        status_auto = f"AUTO: {amostras_auto}/{LIMITE_AUTO} | proximo em {max(0, INTERVALO_AUTO - (agora - ultimo_capture)):.1f}s"
        cv2.putText(frame, status_auto, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Coletor de Dados", frame)
    tecla = cv2.waitKey(1) & 0xFF

    # S — salvar uma amostra manualmente
    if tecla == ord('s') and resultado.hand_landmarks:
        for pontos_mao in resultado.hand_landmarks:
            features = extrair_features(pontos_mao).tolist()
            dados_coletados.append({"label": nota_atual, "features": features})
            amostras_sessao += 1
            print(f"[MANUAL][{nota_atual}] Amostra {amostras_sessao} salva")

    # A — alternar modo automático
    elif tecla == ord('a'):
        modo_auto     = not modo_auto
        amostras_auto = 0        # reseta contador ao iniciar nova sessão auto
        ultimo_capture = 0.0
        estado = "ATIVADO" if modo_auto else "DESATIVADO"
        print(f"Modo automático {estado}")

    # N — próxima nota
    elif tecla == ord('n'):
        nota_idx = (nota_idx + 1) % len(NOTAS)
        print(f"-> {NOTAS[nota_idx]}")

    # P — nota anterior
    elif tecla == ord('p'):
        nota_idx = (nota_idx - 1) % len(NOTAS)
        print(f"-> {NOTAS[nota_idx]}")

    # T — treinar agora
    elif tecla == ord('t'):
        with open(ARQUIVO_DADOS, "w") as f:
            json.dump(dados_coletados, f)
        treinar()

    # Q — sair e salvar
    elif tecla == ord('q'):
        break

with open(ARQUIVO_DADOS, "w") as f:
    json.dump(dados_coletados, f)
print(f"\n{amostras_sessao} novas amostras salvas. Total: {len(dados_coletados)}")

cap.release()
cv2.destroyAllWindows()
detector.close()
