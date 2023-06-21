import cv2
import mediapipe as mp
import pyautogui
import time
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

    image_width, image_height = image.shape[1], image.shape[0]

    # 人差し指(指先)の座標値(x,y)をカメラでキャプチャした画像に合わせる
    index_finger_tip_x = int(hand_landmarks.landmark[8].x * image_width)
    index_finger_tip_y = int(hand_landmarks.landmark[8].y * image_height)

    # 人差し指(第二関節)の座標値(x,y)をカメラでキャプチャした画像に合わせる
    index_finger_pip_x = int(hand_landmarks.landmark[6].x * image_width)
    index_finger_pip_y = int(hand_landmarks.landmark[6].y * image_height)


    # 人差し指を曲げたとき、ダブルクリックをする
    if index_finger_tip_y > index_finger_pip_y:
        pyautogui.doubleClick(index_finger_pip_x,index_finger_pip_y)
        time.sleep(1)                        
                        
    # 上記外は、カーソルを移動させる
    else:
        pyautogui.moveTo(index_finger_pip_x,index_finger_pip_y)

cap.release()

