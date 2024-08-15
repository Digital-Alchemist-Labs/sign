import streamlit as st

def main():
    # CSS 스타일 추가
    st.markdown(
        """
        <style>
        .translateText {
            border: 2px solid black; /* 원하는 색상으로 변경 */
            border-radius: 15px; /* 모서리를 둥글게 */
            padding: 10px;
            margin-bottom: 10px;
        }
        .st-video-container video {
            width: 100% !important;
            height: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('SignGPT')

    # 레이아웃을 두 개의 컬럼으로 나누기
    col1, col2 = st.columns(2)

    # 사진과 동영상의 크기를 동일하게 설정
    media_width = 400  # 이미지 및 동영상의 넓이 설정
    media_height = 300  # 이미지 및 동영상의 높이 설정


    with col2:
        st.header("수어 출력")
        video_url = r'C:\Users\kyo\Desktop\복사영상/춥다11.mov'
        st.video(video_url, start_time=0)
        st.markdown('<div class="translateText"> 번역된 수화 문장 </div>', unsafe_allow_html=True)
        if st.button('번역', use_container_width=True):
            st.session_state.translated = True
        else:
            st.session_state.translated = False

    # 열을 사용하여 레이아웃 설정
    row1, row2 = st.columns([3, 1])

    # 첫 번째 열에 텍스트 필드 추가
    with row1:
        text_input = st.text_input("텍스트로 질문하기")

    # 두 번째 열에 버튼 추가
    with row2:
        # 위쪽 여백
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        if st.button("예시 작동", use_container_width=True):
            st.write(f"입력된 텍스트: {text_input}")

if __name__ == '__main__':
    main()
