import streamlit as st

# Streamlit 웹 애플리케이션 제목
st.title("Streamlit MP4 Video Player")

# 버튼을 생성하여 동영상 재생 트리거
if st.button('Play Video'):
    # 동영상 파일 경로
    video_file = open('output.mp4', 'rb')
    video_bytes = video_file.read()

    # 동영상을 웹 애플리케이션에 표시
    st.video(video_bytes)

# 추가적인 설명 텍스트
st.write("Click the button above to play the video.")
