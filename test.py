import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

# Streamlit 앱의 기본 설정
st.set_page_config(layout="wide")

# 페이지 제목 추가 및 부제목
st.title("SignGPT")
st.markdown("### by . Digital Alchemist")  # 부제목 추가

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

# 오른쪽 컬럼: 라벨 출력 및 영상 재생 버튼
with right_column:
    st.header("결과")
    play_button = st.empty()  # 재생 버튼을 위한 공간
    video_display = st.empty()  # 영상을 표시할 공간
    translated_label = st.empty()

# 시작 버튼이 눌리면 실행
if start_button:
    # cv2 및 Mediapipe 관련 코드
    video_file = "./dataset/안녕하세요.mov"
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hands_keypoints = []

    with mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # Streamlit을 통해 실시간 영상 출력
            video_placeholder.image(image, channels="RGB")

        cap.release()
        cv2.destroyAllWindows()

    # 하드코딩된 JSON 응답 데이터
    hardcoded_response = {
        "words": ["안녕하세요", "무엇", "돕다", "의문"]
    }
    st.session_state.translated_words = hardcoded_response["words"]

    # 단어에 매핑된 수어 영상 파일 경로
    dataset_path = "./dataset"
    video_files = []

    for word in st.session_state.translated_words:
        video_file_path = os.path.join(dataset_path, f"{word}.mov")
        if os.path.exists(video_file_path):
            video_files.append(video_file_path)
        else:
            st.warning(f"Video for word '{word}' not found.")

    # 영상 파일을 합쳐서 보여주는 코드
    output_video = "output.mov"
    if video_files:
        fourcc = cv2.VideoWriter_fourcc(*"mov")
        out = None
        frame_written = False  # 프레임이 작성되었는지 확인하는 플래그

        for video_file in video_files:
            cap = cv2.VideoCapture(video_file)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if out is None:
                    height, width, _ = frame.shape
                    out = cv2.VideoWriter(output_video, fourcc, 20.0, (width, height))

                out.write(frame)
                frame_written = True  # 프레임이 성공적으로 작성됨

            cap.release()

        if out:
            out.release()

    # 영상 파일이 생성된 경우에만 재생 버튼 및 영상을 표시
    if frame_written and os.path.exists(output_video):
        # "재생" 버튼을 표시하고, 누르면 영상 재생
        if play_button.button("재생"):
            video_display.video(output_video)

        # 영상 출력 (항상 표시)
        video_display.video(output_video)

    # JSON 응답 텍스트를 영상 아래에 표시
    translated_label.write(", ".join(st.session_state.translated_words))
