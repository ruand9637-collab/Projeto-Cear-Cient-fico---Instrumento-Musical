import cv2
import mediapipe as mp
import serial
import time
import math

# --- Configuração Serial (Arduino) ---
try:
    arduino = serial.Serial('COM9', 9600, timeout=0.1)
    time.sleep(2)
    print("Arduino conectado!")
except:
    print("Arduino não encontrado. Rodando em modo de simulação visual.")
    arduino = None

# --- Inicialização da IA do MediaPipe ---
mp_maos = mp.solutions.hands
maos = mp_maos.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_desenho = mp.solutions.drawing_utils

# Função auxiliar para calcular distância entre dois pontos da mão
def calcular_distancia(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    sucesso, frame = cap.read()
    if not (determinar := sucesso):
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = maos.process(frame_rgb)

    nota_detectada = "Silencio"

    if resultado.multi_hand_landmarks:
        for pontos_mao in resultado.multi_hand_landmarks:
            # Desenha a malha da mão na tela de forma elegante
            mp_desenho.draw_landmarks(frame, pontos_mao, mp_maos.HAND_CONNECTIONS,
                                     mp_desenho.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                                     mp_desenho.DrawingSpec(color=(0,255,0), thickness=2))

            p = pontos_mao.landmark

            # --- LÓGICA DE DETECÇÃO DE LIBRAS (A até G) ---

            # Verificar se dedos estão esticados (ponta mais alta que a articulação do meio)
            indicador_esticado = p[8].y < p[6].y
            medio_esticado = p[12].y < p[10].y
            anelar_esticado = p[16].y < p[14].y
            minimo_esticado = p[20].y < p[18].y

            # Distâncias importantes para gestos específicos
            dist_polegar_indicador = calcular_distancia(p[4], p[8])
            dist_indicador_medio = calcular_distancia(p[8], p[12])

            # 1. Letra B (Nota Ré) - Todos esticados
            if indicador_esticado and medio_esticado and anelar_esticado and minimo_esticado:
                nota_detectada = "Re (D)"

            # 2. Letra D (Nota Fá) - Apenas indicador esticado para cima
            elif indicador_esticado and not medio_esticado and not anelar_esticado and not minimo_esticado:
                # Garante que não é a letra F ou G checando proximidade do polegar
                if dist_polegar_indicador > 0.1:
                    nota_detectada = "Fa (F)"
                else:
                    nota_detectada = "La (A)"

            # 3. Letra G (Nota Si) - Indicador esticado e Polegar para cima (Y do polegar menor que a base)
            elif indicador_esticado and p[4].y < p[3].y and not medio_esticado:
                nota_detectada = "Si (B)"

            # 4. Letra A (Nota Dó) - Todos fechados, polegar ao lado
            elif not indicador_esticado and not medio_esticado and not anelar_esticado and not minimo_esticado:
                # Se o polegar estiver acima dos dedos dobrados é a letra E (Sol), se for do lado é A (Dó)
                if p[4].y < p[10].y:
                    nota_detectada = "Sol (G)"
                else:
                    # Letra C (Mi) possui formato curvado, testamos pela distância horizontal interna
                    if abs(p[4].x - p[8].x) > 0.15:
                        nota_detectada = "Mi (E)"
                    else:
                        nota_detectada = "Do (C)"

    # Exibe a nota identificada na tela com um design bonito
    cv2.rectangle(frame, (10, 20), (350, 90), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f'NOTA: {nota_detectada}', (20, 70),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

    # Envia o comando para o Arduino
    if arduino:
        # Envia apenas a primeira letra da nota ('A', 'B', 'C'...)
        arduino.write(nota_detectada[0].encode())

    cv2.imshow("Libras Musical AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
if arduino: arduino.close()