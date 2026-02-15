import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO
import speech_recognition as sr
import threading
from pynput.keyboard import Key, Listener
import os

PROJECT_DIR = r"C:\AllsenseML1.0"
MODEL_PATH = os.path.join(PROJECT_DIR, "best.pt")
FACE_CASCADE_PATH = os.path.join(PROJECT_DIR, "haarcascade_frontalface_default.xml")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"model not fnd: {MODEL_PATH}")
if not os.path.exists(FACE_CASCADE_PATH):
    raise FileNotFoundError(f"Cascade fnd: {FACE_CASCADE_PATH}")

pyautogui.FAILSAFE = False
FPS_LIMIT = 30
IMG_SIZE = 320
SENSITIVITY = 30.0 
SPEED = 0.08 
SMOOTHING = 0.95 
REST_ZONE = 60

model = YOLO(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError("XML Er")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, FPS_LIMIT)
blink_counter = 0
blinked = False
is_speaking = False
prev_x, prev_y = pyautogui.position()
def voice_input():
    global is_speaking
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            is_speaking = True
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio, language="ru-RU")
            pyautogui.typewrite(text + " ")
    except:
        pass
    finally:
        is_speaking = False

def on_press(key):
    if key == Key.f2 and not is_speaking:
        threading.Thread(target=voice_input, daemon=True).start()

listener = Listener(on_press=on_press)
listener.start()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        cv2.putText(frame, "User", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    results = model(frame, verbose=False, imgsz=IMG_SIZE)
    pupils = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            if conf < 0.4 or cls != 0:
                continue
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            pupils.append((cx, cy))
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

    if pupils:
        blinked = False
        blink_counter = 0
        cx, cy = pupils[0]

        if abs(cx - center_x) < REST_ZONE and abs(cy - center_y) < REST_ZONE:
            pass
        else:
            screen_w, screen_h = pyautogui.size()
            target_x = np.interp(cx, [0, w], [0, screen_w])
            target_y = np.interp(cy, [0, h], [0, screen_h])

            current_x, current_y = pyautogui.position()
            dx = (target_x - current_x) * SENSITIVITY
            dy = (target_y - current_y) * SENSITIVITY

            smooth_x = prev_x * SMOOTHING + (current_x + dx) * (1 - SMOOTHING)
            smooth_y = prev_y * SMOOTHING + (current_y + dy) * (1 - SMOOTHING)

            pyautogui.moveTo(smooth_x, smooth_y, duration=SPEED)
            prev_x, prev_y = smooth_x, smooth_y

    elif not blinked:
        blink_counter += 1
        if blink_counter >= 2:
            pyautogui.click()
            blinked = True
            blink_counter = 0

    cv2.imshow("AllSense AAC", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
listener.stop()