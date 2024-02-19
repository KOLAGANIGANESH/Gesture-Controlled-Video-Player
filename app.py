from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pyautogui
import time

app = Flask(__name__)



def count_fingers(lst):
    cnt = 0

    threshold = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2

    if lst.landmark[5].y * 100 - lst.landmark[8].y * 100 > threshold:
        cnt += 1
    if lst.landmark[9].y * 100 - lst.landmark[12].y * 100 > threshold:
        cnt += 1
    if lst.landmark[13].y * 100 - lst.landmark[16].y * 100 > threshold:
        cnt += 1
    if lst.landmark[17].y * 100 - lst.landmark[20].y * 100 > threshold:
        cnt += 1
    if lst.landmark[5].x * 100 - lst.landmark[4].x * 100 > 6:
        cnt += 1

    return cnt

def hand_gesture_recognition():
    cap = cv2.VideoCapture(0)
    drawing = mp.solutions.drawing_utils
    hands = mp.solutions.hands.Hands(max_num_hands=1)

    prev_cnt = -1
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            current_cnt = count_fingers(hand_landmarks)

            if current_cnt != prev_cnt:
                elapsed_time = time.time() - start_time
                if elapsed_time > 0.2:
                    if current_cnt == 1:
                        pyautogui.press("right")
                    elif current_cnt == 2:
                        pyautogui.press("left")
                    elif current_cnt == 3:
                        pyautogui.press("up")
                    elif current_cnt == 4:
                        pyautogui.press("down")
                    elif current_cnt == 5:
                        pyautogui.press("space")
                    start_time = time.time()
                    prev_cnt = current_cnt

            drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(hand_gesture_recognition(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
