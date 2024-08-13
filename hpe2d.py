import mediapipe as mp
import cv2
import numpy as np
from model_dim import InferModel
import torch
import yaml
from utils import *
from PIL import ImageFont, ImageDraw, Image
import warnings
import json
import requests

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    with open("configs/default.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    num_classes = cfg["num_classes"]
    ckpt_name = cfg["ckpt_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)  

    w = 640
    h = 480

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    ui_img = make_ui_img(w)

    hands_keypoints = []
    translation_flag = False
    korean = [
        "행복",
        "안녕",
        "슬픔",
        "눈",
        "당신",
        "식사하셨어요?",
        "이름이 뭐 예요?",
        "사랑",
        "수어",
        "만나서 반가워요!",
        "끝"  # '끝'을 추가하여 번역 종료 시점을 감지합니다.
    ]
    model = InferModel.load_from_checkpoint(checkpoint_path=ckpt_name)
    model.eval().to(device)  
    print("Model loaded and moved to device successfully")
    label_play_time = 0
    display_flag = False

    translated_words = []  # 번역된 단어를 저장할 리스트

    with mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image from camera")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                if translation_flag and len(results.multi_handedness) == 2:
                    keypoints_on_frame = []
                    left_hand, right_hand = False, False
                    for num_hand, hand_type in enumerate(results.multi_handedness):
                        hand_score = hand_type.classification[0].score
                        hand_type = hand_type.classification[0].index
                        if hand_score < 0.8:
                            continue
                        if int(hand_type) == 0:
                            left_hand = True
                        if int(hand_type) == 1:
                            right_hand = True
                        keypoints_on_frame.extend(
                            landmarkxy2list(results.multi_hand_landmarks[num_hand])
                        )
                    if left_hand and right_hand:
                        hands_keypoints.append(keypoints_on_frame)

                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

            image = drawUI(image, ui_img, np.zeros((100, w, 3), np.uint8))

            if display_flag:
                image = display_label(labels, image, w, h)
                label_play_time += 1

            if label_play_time > 150:
                label_play_time = 0
                display_flag = False

            cv2.imshow("Sign-Language Translator", image)

            key_queue = cv2.waitKey(1)
            if (key_queue & 0xFF == ord("s")) and not translation_flag:
                print("Start recording")
                translation_flag = True
                key_queue = 0

            if (key_queue & 0xFF == ord("e")) and translation_flag:
                print("Save record...")
                translation_flag = False
                print("End recording")
                key_queue = 0
                hands_keypoints = torch.tensor(hands_keypoints)
                frames_len = hands_keypoints.shape[0]
                ids = np.round(np.linspace(0, frames_len - 1, 60))
                keypoint_sequence = []
                for i in range(60):
                    keypoint_sequence.append(
                        hands_keypoints[int(ids[i]), ...].unsqueeze(0)
                    )
                keypoint_sequence = torch.cat(keypoint_sequence, dim=0)
                input_data = keypoint_sequence.unsqueeze(0).to(device)
                print(keypoint_sequence.shape)
                output = model(input_data)
                labels = torch.max(output, dim=1)[1][0]
                labels = korean[labels]

                # 번역된 단어를 리스트에 추가
                translated_words.append(labels)

                # '끝'이라는 단어가 감지되면 모아둔 단어들을 JSON 형식으로 변환하고 전송
                if labels == "끝":
                    # JSON 형식으로 변환
                    json_data = json.dumps({"words": translated_words})

                    # 전송할 서버의 IP 주소 및 포트 번호
                    server_ip = "http://your-server-ip-address:your-port-number"

                    # POST 요청을 통해 데이터 전송
                    try:
                        response = requests.post(server_ip, data=json_data, headers={"Content-Type": "application/json"})
                        if response.status_code == 200:
                            print("Data sent successfully!")
                        else:
                            print(f"Failed to send data, status code: {response.status_code}")
                    except Exception as e:
                        print(f"Error sending data: {e}")

                    # 단어 리스트 초기화
                    translated_words = []

                display_flag = True
                hands_keypoints = []

            if key_queue & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
