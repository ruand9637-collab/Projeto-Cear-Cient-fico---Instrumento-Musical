import cv2
import os
import mediapipe as mp
import serial
import time
import math
from mediapipe.tasks import python
# from mediapipe.tasclks.python import vision
from mediapipe.tasks.python import vision
from pygrabber.dshow_graph import FilterGraph

# =========================================================================
# CONFIGURAÇÃO DE DISPOSITIVOS
# =========================================================================
NOME_DA_CAMERA_DESEJADA = "OBS Virtual Camera"
# "DroidCam Source 2"
PORTA_SERIAL = 'COM13'
BAUD_RATE = 115200  

def buscar_indice_camera_por_nome(nome_procurado):
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

# --- Configuração Serial ---
try:
    arduino = serial.Serial(PORTA_SERIAL, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    print("Arduino conectado com sucesso em 115200 baud!")
except Exception as e:
    print(f"Erro ao conectar no Arduino: {e}")
    print("Rodando em modo de simulação visual.")
    arduino = None

# --- Inicialização da IA ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,  
    min_hand_detection_confidence=0.5,  
    min_hand_presence_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

indice_camera = buscar_indice_camera_por_nome(NOME_DA_CAMERA_DESEJADA)
cap = cv2.VideoCapture(indice_camera, cv2.CAP_DSHOW)

print("Aguardando inicialização da DroidCam...")
time.sleep(1.0)

# --- VARIÁVEIS DE CONTROLE E HISTÓRICO ---
ultima_nota = ""
y_pulso_anterior = 0.5  
mao_solo_detectada_anteriormente = False  

# --- FILTROS DE AMORTECIMENTO (EMA) ---
FATOR_SUAVIZACAO = 0.35  
pos_suave_y = {"Left": 0.5, "Right": 0.5}
pos_suave_z = {"Left": 0.0, "Right": 0.0}

# --- VARIÁVEIS DE ESTADO DOS INSTRUMENTOS ---
instrumentos = ["TECLADO", "VIOLINO", "VIOLAO"]
prefixos_serial = ["T", "V", "A"] 
idx_instrumento = 0

tempo_inicio_punho = None  
trocou_instrumento = False 

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    resultado = detector.detect(mp_image)

    nota_solo, vibrato_solo, oitava_solo = "X", 0, 2
    acorde_base = "X"
    punho_esquerda, punho_direita = False, False
    ataque_solo = 0  
    solo_encontrado_neste_frame = False  

    if resultado.hand_landmarks:
        for idx, pontos_mao in enumerate(resultado.hand_landmarks):
            info_lado = resultado.handedness[idx][0]
            lado_real = info_lado.category_name  

            if lado_real == "Left":
                cor_mao = (255, 150, 0)   
            else:
                cor_mao = (0, 0, 255)     

            for ponto in pontos_mao:
                cx, cy = int(ponto.x * w), int(ponto.y * h)
                cv2.circle(frame, (cx, cy), 3, cor_mao, -1)

            p = pontos_mao

            # --- DETECÇÃO DE DEDOS ---
            indicador_esticado = p[8].y < p[6].y
            medio_esticado = p[12].y < p[10].y
            anelar_esticado = p[16].y < p[14].y
            minimo_esticado = p[20].y < p[18].y
            polegar_esticado = abs(p[4].x - p[17].x) > abs(p[2].x - p[17].x)

            # --- VERIFICAÇÃO SE É UM PUNHO FECHADO ---
            punho_fechado = not indicador_esticado and not medio_esticado and not anelar_esticado and not minimo_esticado and not polegar_esticado
            if lado_real == "Right": punho_esquerda = punho_fechado
            elif lado_real == "Left": punho_direita = punho_fechado

            # --- CLASSIFICAÇÃO DOS GESTOS ---
            gesto_atual = "X"
            if indicador_esticado and not medio_esticado and not anelar_esticado and not minimo_esticado and not polegar_esticado:
                gesto_atual = "C"  
            elif indicador_esticado and medio_esticado and not anelar_esticado and not minimo_esticado and not polegar_esticado:
                gesto_atual = "D"  
            elif indicador_esticado and medio_esticado and anelar_esticado and not minimo_esticado and not polegar_esticado:
                gesto_atual = "E"  
            elif indicador_esticado and medio_esticado and anelar_esticado and minimo_esticado and not polegar_esticado:
                gesto_atual = "F"  
            elif indicador_esticado and medio_esticado and anelar_esticado and minimo_esticado and polegar_esticado:
                gesto_atual = "G"  
            elif polegar_esticado and minimo_esticado and not indicador_esticado and not medio_esticado and not anelar_esticado:
                gesto_atual = "A"  
            elif minimo_esticado and not indicador_esticado and not medio_esticado and not anelar_esticado and not polegar_esticado:
                gesto_atual = "B"  

            # --- PROCESSAMENTO FILTRADO (Mão do Solo / Direita Física) ---
            if lado_real == "Left":
                nota_solo = gesto_atual
                solo_encontrado_neste_frame = True

                # Gatilho de velocidade direcional (Para baixo)
                y_pulso_atual = p[0].y
                if mao_solo_detectada_anteriormente:
                    velocidade_deslocamento = y_pulso_atual - y_pulso_anterior
                    if velocidade_deslocamento > 0.035:  
                        ataque_solo = 1
                else:
                    mao_solo_detectada_anteriormente = True

                y_pulso_anterior = y_pulso_atual  

                # Filtro EMA - Eixo Y (Vibrato)
                y_cru = p[0].y
                y_filtrado = (FATOR_SUAVIZACAO * y_cru) + ((1 - FATOR_SUAVIZACAO) * pos_suave_y[lado_real])

                delta_y = abs(y_filtrado - pos_suave_y[lado_real])
                pos_suave_y[lado_real] = y_filtrado  

                if delta_y > 0.005:  
                    vibrato_solo = min(int(delta_y * 400), 7)  

                # Filtro EMA - Eixo Z (Oitava)
                z_cru = p[0].z
                z_filtrado = (FATOR_SUAVIZACAO * z_cru) + ((1 - FATOR_SUAVIZACAO) * pos_suave_z[lado_real])
                pos_suave_z[lado_real] = z_filtrado

                if z_filtrado < -0.06: oitava_solo = 3  
                elif z_filtrado > 0.04: oitava_solo = 1  
                else: oitava_solo = 2  

            elif lado_real == "Right":
                acorde_base = gesto_atual

    if not solo_encontrado_neste_frame:
        mao_solo_detectada_anteriormente = False

    # --- LÓGICA DE TROCA DE INSTRUMENTO POR TEMPO ---
    if len(resultado.hand_landmarks) == 2 and punho_esquerda and punho_direita:
        if tempo_inicio_punho is None:
            tempo_inicio_punho = time.time()  

        tempo_decorrido = time.time() - tempo_inicio_punho
        porcentagem = min(tempo_decorrido / 1.5, 1.0)
        cv2.rectangle(frame, (int(w/2 - 100), int(h - 30)), (int(w/2 - 100 + (200 * porcentagem)), int(h - 15)), (0, 255, 255), cv2.FILLED)

        if tempo_decorrido > 1.5 and not trocou_instrumento:
            idx_instrumento = (idx_instrumento + 1) % len(instrumentos)
            trocou_instrumento = True  
    else:
        tempo_inicio_punho = None   
        trocou_instrumento = False  

    # --- INTERFACE VISUAL ---
    cv2.rectangle(frame, (10, 10), (630, 105), (0, 0, 0), cv2.FILLED)
    cores_interface = [(255, 255, 255), (0, 255, 0), (255, 128, 0)] 
    cor_atual = cores_interface[idx_instrumento]

    cv2.putText(frame, f"INST: {instrumentos[idx_instrumento]}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, cor_atual, 2)

    texto_atk = " | *GOLPE*" if ataque_solo == 1 else ""
    texto_feedback = f"SOLO (M.Dir): {nota_solo}(V:{vibrato_solo} O:{oitava_solo}){texto_atk} | BASE (M.Esq): {acorde_base}"
    cv2.putText(frame, texto_feedback, (20, 85), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(frame, f"OITAVA ATUAL: {oitava_solo}", (int(w - 180), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_atual, 1)

    # --- OSCILOSCÓPIO VISUAL ---
    tempo_animacao = time.time() * 30
    amplitude = 30 if (nota_solo != "X" or acorde_base != "X") else 3
    frequencias_visuais = {"C": 0.04, "D": 0.06, "E": 0.08, "F": 0.10, "G": 0.12, "A": 0.14, "B": 0.16, "X": 0.02}

    nota_prioritaria = nota_solo if nota_solo != "X" else acorde_base
    freq_onda = frequencias_visuais.get(nota_prioritaria, 0.02)
    fator_vibrato = (vibrato_solo * 0.015) if vibrato_solo > 0 else 0

    for x in range(20, w - 20, 4):
        y = int((h - 50) + amplitude * math.sin(x * (freq_onda + fator_vibrato) + tempo_animacao))
        cv2.circle(frame, (x, y), 2, cor_atual, -1) 

    # --- PROTOCOLO SERIAL ---
    ins_atual = prefixos_serial[idx_instrumento]
    comando = f"{nota_solo},{vibrato_solo},{oitava_solo},{acorde_base},{ins_atual},{ataque_solo}\n"

    if arduino and (comando != ultima_nota or ataque_solo == 1):
        try:
            arduino.write(comando.encode())
            print(f"Enviado para o ESP32: {comando.strip()}")
            ultima_nota = comando
        except Exception as e:
            print(f"Erro durante o envio: {e}")
            arduino = None

    cv2.imshow("Libras Musicais", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino: arduino.close()
detector.close()