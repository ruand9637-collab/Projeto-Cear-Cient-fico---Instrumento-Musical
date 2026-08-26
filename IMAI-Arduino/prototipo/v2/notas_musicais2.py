import cv2
import mediapipe as mp
import serial
import time
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pygrabber.dshow_graph import FilterGraph  # Mapeia os nomes das câmeras

# =========================================================================
# CONFIGURAÇÃO DE DISPOSITIVOS
# =========================================================================
NOME_DA_CAMERA_DESEJADA = "OBS Virtual Camera"  # Nome padrão do OBS no Windows
PORTA_SERIAL = 'COM9'


# =========================================================================

def buscar_indice_camera_por_nome(nome_procurado):
    """Varre o sistema operacional buscando o índice correto da câmera pelo nome."""
    graph = FilterGraph()
    dispositivos = graph.get_input_devices()

    print("\n--- Câmeras detectadas no sistema ---")
    for idx, nome in enumerate(dispositivos):
        print(f"Índice [{idx}]: {nome}")
        if nome_procurado.lower() in nome.lower():
            print(f"-> Sucesso! Encontrado '{nome}' no Índice {idx}\n")
            return idx

    print(f"-> Alerta: '{nome_procurado}' não foi encontrado. Usando câmera padrão [0].\n")
    return 0


# --- Configuração Serial (Arduino) ---
try:
    arduino = serial.Serial(PORTA_SERIAL, 9600, timeout=0.1)
    time.sleep(2)  # Tempo essencial para o Arduino resetar ao conectar
    print("Arduino conectado com sucesso!")
except Exception as e:
    print(f"Erro ao conectar no Arduino: {e}")
    print("Rodando em modo de simulação visual.")
    arduino = None

# --- Inicialização da NOVA IA do MediaPipe (tasks) ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8
)
detector = vision.HandLandmarker.create_from_options(options)


def calcular_distancia(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


# --- CAPTURA DA CÂMERA PELO NOME ---
indice_camera = buscar_indice_camera_por_nome(NOME_DA_CAMERA_DESEJADA)

# Abre a câmera usando o DirectShow e o índice dinâmico encontrado
cap = cv2.VideoCapture(indice_camera, cv2.CAP_DSHOW)

# --- AJUSTE DE INICIALIZAÇÃO DA VIRTUAL CAM ---
print("Aguardando inicialização da Câmera Virtual...")
time.sleep(1.5)

# Força o OpenCV a tentar ler frames iniciais caso o OBS demore a responder
for tentativa in range(5):
    sucesso, frame = cap.read()
    if sucesso:
        print("Fluxo de vídeo do OBS recebido com sucesso!")
        break
    print(f"Tentando conectar ao fluxo do OBS... ({tentativa + 1}/5)")
    time.sleep(0.5)
# ==========================================

# Variável de controle para NÃO inundar o Arduino de dados repetidos
ultima_nota = ""

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    resultado = detector.detect(mp_image)
    nota_detectada = "Silencio"

    if resultado.hand_landmarks:
        for pontos_mao in resultado.hand_landmarks:

            # --- DESENHO DOS PONTOS ---
            for ponto in pontos_mao:
                cx, cy = int(ponto.x * w), int(ponto.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            p = pontos_mao

            # --- LÓGICA DE DETECÇÃO ---
            indicador_esticado = p[8].y < p[6].y
            medio_esticado = p[12].y < p[10].y
            anelar_esticado = p[16].y < p[14].y
            minimo_esticado = p[20].y < p[18].y

            dist_polegar_indicador = calcular_distancia(p[4], p[8])

            if indicador_esticado and medio_esticado and anelar_esticado and minimo_esticado:
                nota_detectada = "Re (D)"
            elif indicador_esticado and not medio_esticado and not anelar_esticado and not minimo_esticado:
                if dist_polegar_indicador > 0.1:
                    nota_detectada = "Fa (F)"
                else:
                    nota_detectada = "La (A)"
            elif indicador_esticado and p[4].y < p[3].y and not medio_esticado:
                nota_detectada = "Si (B)"
            elif not indicador_esticado and not medio_esticado and not anelar_esticado and not minimo_esticado:
                if p[4].y < p[10].y:
                    nota_detectada = "Sol (G)"
                else:
                    if abs(p[4].x - p[8].x) > 0.15:
                        nota_detectada = "Mi (E)"
                    else:
                        nota_detectada = "Do (C)"

    # Exibe o painel na tela
    cv2.rectangle(frame, (10, 20), (350, 90), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f'NOTA: {nota_detectada}', (20, 70),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

    # --- ENVIO INTELIGENTE PARA O ARDUINO ---
    if arduino and nota_detectada != ultima_nota:
        # === ALTERAÇÃO INSERIDA AQUI ===
        if nota_detectada == "Silencio":
            comando = "X\n"
        else:
            # Pega a letra que está dentro dos parênteses (ex: 'C' de Do (C))
            letra_cifra = nota_detectada.split('(')[1][0]
            comando = f"{letra_cifra}\n"
        # ===============================

        try:
            arduino.write(comando.encode())
            print(f"Enviado para o Arduino: {comando.strip()}")
            ultima_nota = nota_detectada
        except Exception as e:
            print(f"Erro durante o envio: {e}")
            arduino = None

    cv2.imshow("Libras Musical AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
detector.close()