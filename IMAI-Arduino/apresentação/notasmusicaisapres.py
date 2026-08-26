import cv2
import mediapipe as mp
import serial
import time
import math

# --- CONFIGURAÇÃO SERIAL (ARDUINO) ---
try:
    arduino = serial.Serial('COM3', 9600, timeout=0.1)
    time.sleep(2)  # Tempo para o Arduino resetar
    print("Arduino conectado com sucesso!")
except:
    print("Arduino não encontrado. Rodando em modo de simulação.")
    arduino = None

# --- CLASSE DE DETECÇÃO ---
class DetectorMaos:
    def __init__(self, max_maos=1, deteccao_confianca=0.8, rastreio_confianca=0.8):
        self.mp_maos = mp.solutions.hands
        self.maos = self.mp_maos.Hands(
            max_num_hands=max_maos,
            min_detection_confidence=deteccao_confianca,
            min_tracking_confidence=rastreio_confianca
        )
        self.mp_desenho = mp.solutions.drawing_utils

    def processar(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = self.maos.process(frame_rgb)
        return resultado

    def desenhar(self, frame, resultado):
        if resultado.multi_hand_landmarks:
            for pontos_mao in resultado.multi_hand_landmarks:
                self.mp_desenho.draw_landmarks(
                    frame, pontos_mao, self.mp_maos.HAND_CONNECTIONS,
                    self.mp_desenho.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    self.mp_desenho.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

# --- FUNÇÕES AUXILIARES ---
def calcular_distancia(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# --- INICIALIZAÇÃO ---
detector = DetectorMaos()
cap = cv2.VideoCapture(0)

# Variáveis de Estabilidade (Debounce)
nota_anterior = "Silencio"
contador_estabilidade = 0
LIMITE_ESTABILIDADE = 5  # Quantos frames iguais para validar a nota

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso: break

    frame = cv2.flip(frame, 1)
    resultado = detector.processar(frame)
    detector.desenhar(frame, resultado)

    nota_atual = "Silencio"

    if resultado.multi_hand_landmarks:
        for pontos_mao in resultado.multi_hand_landmarks:
            p = pontos_mao.landmark
            
            # Lógica dos dedos (y diminui conforme sobe na tela)
            ind_est = p[8].y < p[6].y
            med_est = p[12].y < p[10].y
            ane_est = p[16].y < p[14].y
            min_est = p[20].y < p[18].y

            # Lógica das Notas
            if ind_est and med_est and ane_est and min_est:
                nota_atual = "Re"
            elif ind_est and not med_est and not ane_est and not min_est:
                if calcular_distancia(p[4], p[8]) > 0.1: 
                    nota_atual = "Fa"
                else:
                    nota_atual = "La"
            elif ind_est and p[4].y < p[3].y and not med_est:
                nota_atual = "Si"
            elif not ind_est and not med_est and not ane_est and not min_est:
                if p[4].y < p[10].y:
                    nota_atual = "Sol"
                elif abs(p[4].x - p[8].x) > 0.15:
                    nota_atual = "Mi"
                else:
                    nota_atual = "Do"

    # --- SISTEMA DE ESTABILIZAÇÃO (DEBOUNCE) ---
    if nota_atual == nota_anterior:
        contador_estabilidade += 1
    else:
        contador_estabilidade = 0
        nota_anterior = nota_atual

    # Envia apenas se estabilizar
    if contador_estabilidade == LIMITE_ESTABILIDADE:
        if arduino and nota_atual != "Silencio":
            # Envia a primeira letra da nota (D, R, M, F, S, L, S)
            arduino.write(nota_atual[0].encode())
            print(f"Nota enviada: {nota_atual}")

    # Interface Visual
    cv2.rectangle(frame, (10, 20), (300, 90), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f'NOTA: {nota_atual}', (20, 70), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

    cv2.imshow("Libras Musical AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
if arduino: arduino.close()