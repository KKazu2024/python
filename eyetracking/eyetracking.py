import cv2
import dlib
import numpy as np
from imutils import face_utils
import pyautogui

# 顔検出器を初期化し、アイトラッキング用の特定の特徴点のインデックスを取得する
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# pyautogui.FAILSAFE = False

# 画面のサイズを取得する
screenWidth, screenHeight = pyautogui.size()

# カメラからのビデオストリームを開始する
cap = cv2.VideoCapture(0)

# 初期化
leftEyeClosed = False
leftEyeClosedCounter = 0

while True:
    # フレームを取得する
    ret, frame = cap.read()
    if not ret:
        break

    # 顔を検出する
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    # 検出された各顔に対してアイトラッキングを実行する
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # 左目と右目の瞳孔の中心を取得する
        leftEyeCenter = leftEye.mean(axis=0).astype("int")
        rightEyeCenter = rightEye.mean(axis=0).astype("int")

        # 瞳孔の中心の座標をスクリーン座標に変換する
        x = int((leftEyeCenter[0] + rightEyeCenter[0]) / 2)
        y = int((leftEyeCenter[1] + rightEyeCenter[1]) / 2)
        # screenX = int((x / frame.shape[1]) * screenWidth)
        screenX = int(((1 - x / frame.shape[1]) * screenWidth))
        screenY = int((y / frame.shape[0]) * screenHeight)
        
        # 画面の端に達した場合に、マウスカーソルを反対側の端に移動する
        if screenX >= screenWidth:
            screenX = screenWidth - 1
        elif screenX < 0:
            screenX = 0
        if screenY >= screenHeight:
            screenY = screenHeight - 1
        elif screenY < 0:
            screenY = 0

        # マウスカーソルを移動する
        pyautogui.moveTo(screenX, screenY)
        # pyautogui.moveRel(screenX, screenY)

        # 左目が閉じられたときにカウントする
        # if leftEyeClosed:
        #     if leftEye[0][1] > leftEye[3][1]:
        #         leftEyeClosedCounter += 1
        #     else:
        #         leftEyeClosed = False
        # else:
        #     if leftEye[0][1] < leftEye[2][1]:
        #         leftEyeClosed = True

        # 左目が2回閉じられたときに左クリックする
        # if leftEyeClosedCounter == 2:
        #     pyautogui.click(button='left')
        #     leftEyeClosedCounter = 0

    # 結果を表示する
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()