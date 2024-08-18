import streamlit as st
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

# Streamlit 앱의 기본 설정
st.set_page_config(layout="wide")

# 페이지 제목 추가
st.title("SignGPT")

# Mediapipe와 관련된 설정
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 모델 로드 및 초기화
with open("configs/default.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

num_classes = cfg["num_classes"]
ckpt_name = cfg["ckpt_name"]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = InferModel.load_from_checkpoint(checkpoint_path=ckpt_name)
model.eval().to(device)
print("Model loaded and moved to device successfully")

# Korean labels
korean = [
    "행복", "안녕", "슬픔", "눈", "당신", "식사하셨어요?",
    "이름이 뭐 예요?", "사랑", "수어", "만나서 반가워요!",
    "미세먼지", "싫다", "무엇인가요?", "날씨", "끝"
]

# 기존에 번역된 단어를 저장할 리스트 (세션 상태 유지)
if "translated_words" not in st.session_state:
    st.session_state.translated_words = []

# Streamlit의 컬럼 레이아웃 설정
left_column, right_column = st.columns(2)

# 왼쪽 컬럼: 실시간 영상 및 버튼
with left_column:
    st.header("수어 입력")
    start_button = st.button("시작")
    video_placeholder = st.empty()  # 비디오 출력을 위한 공간

# 오른쪽 컬럼: 라벨 출력 (이 부분은 계속 업데이트)
with right_column:
    st.header("결과")
    label_placeholder = st.empty()
    translated_label = st.empty()

    # 번역된 단어 리스트 업데이트
    translated_label.write(f"Translated Label Array: {st.session_state.translated_words}")

# 서버 IP와 포트 번호 설정
server_ip = "http://your-server-ip-address:your-port-number"

# 시작 버튼이 눌리면 실행
if start_button:
    # cv2 및 Mediapipe 관련 코드
    video_file = r"C:\Users\kyo\Desktop\Human Activity Recognition using TensorFlow (CNN + LSTM) Code\dataset\수어\서울/서울1.mov"  # 사용하려는 MP4 파일의 경로를 입력하세요.
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    translation_flag = True

    ui_img = make_ui_img(640)
    hands_keypoints = []
    label_play_time = 0
    display_flag = False

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                if len(results.multi_handedness) == 2:
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
                        keypoints_on_frame.extend(landmarkxy2list(results.multi_hand_landmarks[num_hand]))
                    if left_hand and right_hand:
                        hands_keypoints.append(keypoints_on_frame)

                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

            # drawUI 함수 호출 시 이미지 크기 맞추기
            resized_ui_img = cv2.resize(ui_img, (image.shape[1], 100))
            subtitle_bg = np.zeros_like(resized_ui_img)

            image = drawUI(image, resized_ui_img, subtitle_bg)

            if display_flag:
                image = display_label(labels, image, 640, 480)
                label_play_time += 1

            if label_play_time > 150:
                label_play_time = 0
                display_flag = False

            # Streamlit을 통해 실시간 영상 출력
            video_placeholder.image(image, channels="BGR")

        # 영상이 끝난 후 처리
        if hands_keypoints:
            hands_keypoints = torch.tensor(hands_keypoints)
            frames_len = hands_keypoints.shape[0]

            if frames_len >= 60:
                ids = np.round(np.linspace(0, frames_len - 1, 60)).astype(int)
                keypoint_sequence = []
                for i in range(60):
                    keypoint_sequence.append(hands_keypoints[ids[i], ...].unsqueeze(0))
                keypoint_sequence = torch.cat(keypoint_sequence, dim=0)
                input_data = keypoint_sequence.unsqueeze(0).to(device)
                output = model(input_data)
                labels = torch.max(output, dim=1)[1][0]
                labels = korean[labels]
            else:
                labels = "No valid sequence found"

            # 번역된 단어를 리스트에 추가
            st.session_state.translated_words.append(labels)

            # "끝"이라는 라벨이 추가되면 서버로 전송하고 배열 초기화
            if labels == "끝":
                json_data = json.dumps({"words": st.session_state.translated_words})
                try:
                    response = requests.post(server_ip, data=json_data, headers={"Content-Type": "application/json"})
                    if response.status_code == 200:
                        print("Data sent successfully!")
                    else:
                        print(f"Failed to send data, status code: {response.status_code}")
                except Exception as e:
                    print(f"Error sending data: {e}")

                # 배열 초기화
                st.session_state.translated_words = []

            # Streamlit을 통해 번역된 라벨 출력
            label_placeholder.write(f"Translated Label: {labels}")

    cap.release()
    cv2.destroyAllWindows()
