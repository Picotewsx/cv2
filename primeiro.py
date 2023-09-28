import cv2
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Inicialize o módulo MediaPipe Hands
mp_hands = mp.solutions.hands

# Inicialize o OpenCV
cap = cv2.VideoCapture(0)  # Use 0 para a câmera padrão

# Inicialize o objeto de controle de volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Variável para controlar o volume
previous_hand_state = None

with mp_hands.Hands() as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Converta a imagem para escala de cinza
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processar a imagem para detecção de mãos
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Determine a posição da mão (aberta ou fechada)
                hand_state = "aberta" if hand_landmarks.landmark[4].y < hand_landmarks.landmark[2].y else "fechada"

                # Se o estado da mão mudou, ajuste o volume
                if hand_state != previous_hand_state:
                    if hand_state == "fechada":
                        volume.SetMasterVolumeLevelScalar(1.0, None)  # Aumentar o volume
                    else:
                        volume.SetMasterVolumeLevelScalar(0.0, None)  # Diminuir o volume

                previous_hand_state = hand_state

                # Desenhe os pontos das articulações da mão na imagem
                for point in hand_landmarks.landmark:
                    x, y = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Exiba a imagem com os pontos das articulações da mão
        cv2.imshow("Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()