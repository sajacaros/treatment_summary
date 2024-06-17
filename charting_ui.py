import os
import re

import dotenv
import requests
import streamlit as st
import yaml
from streamlit_authenticator import Authenticate
from yaml import SafeLoader

sts_url = "http://localhost:8090/sts"
charting_url = "http://localhost:8090/charting"


def init_session():
    print('init_session')
    st.session_state.filename = None
    st.session_state.stt_result = None
    st.session_state.stt_method = None
    st.session_state.left_charting = None
    st.session_state.right_charting = None

def call_status_api():
    try:
        res = requests.get(sts_url, timeout=1)
    except:
        return False
    return 200 <= res.status_code < 300


def call_sts_api(sts_method, upload_dir):
    return requests.post(sts_url, json={
        "method": sts_method,
        "filename": st.session_state.filename,
        "upload_dir": upload_dir
    }).json()


def call_charting_api(txt_path, llm, prompt):
    return requests.post(charting_url, json={
        "text_path": txt_path,
        "llm": llm,
        "prompt": prompt
    }).json()

@st.cache_data
def get_datapath():
    dotenv.load_dotenv()
    data_path = os.environ.get('DATA_PATH')
    return data_path

def extract_contents(text):
    contents = {}
    sections = ["요약", "차팅", "SOAP", "근거 정리"]

    # 각 section을 정규표현식으로 분할하여 추출
    pattern = r"=+([^=]+)=+\n([\s\S]*?)(?=\n=+|$)"
    matches = re.findall(pattern, text)

    for i in range(len(matches)):
        section_name = matches[i][0].strip()
        section_content = matches[i][1].strip()

        # section_name이 sections에 포함되어 있으면 contents에 추가
        if section_name in sections:
            contents[section_name] = section_content

    return contents

def creat_llm_result(key):
    conf_llm, conf_prompt, conf_button = st.columns([2, 2, 1])
    with conf_llm:
        selected_llm = st.selectbox('모델 선택', ('GPT4', 'GPT3.5', 'Gemini'), key=f'llm_{key}').lower()
    with conf_prompt:
        selected_prompt = st.selectbox('Prompt 선택', ('Basic', 'Knowledge', 'CoT'), key=f'prompt_{key}').lower()
    with conf_button:
        st.markdown('<div class="centered">', unsafe_allow_html=True)
        if st.button('차팅 실행', key=f'charting_{key}', type='primary', disabled=False if st.session_state.stt_result else True):
            charting_result = call_charting_api(st.session_state.stt_result['text_path'], selected_llm, selected_prompt)
            st.session_state[f'{key}_charting'] = charting_result
    with st.expander('Resource', expanded=False):
        st.write(f"Total Tokens : {st.session_state[f'{key}_charting']['resource']['total_tokens'] if st.session_state[f'{key}_charting'] else 0}")
        st.write(f"Prompt Tokens: {st.session_state[f'{key}_charting']['resource']['prompt_tokens'] if st.session_state[f'{key}_charting'] else 0}")
        st.write(f"Completion Tokens: {st.session_state[f'{key}_charting']['resource']['completion_tokens'] if st.session_state[f'{key}_charting'] else 0}")
        st.write(f"Total Cost (USD): {st.session_state[f'{key}_charting']['resource']['total_cost'] if st.session_state[f'{key}_charting'] else '$ 0.0'}")
    contents=None
    if st.session_state[f'{key}_charting']:
        with open(os.path.join(get_datapath(), st.session_state[f'{key}_charting']['charting_filepath']), 'r', encoding='utf-8') as f:
            contents = extract_contents(f.read())
    tab_charting, tab_summary, tab_soap, tab_reason = st.tabs(["차팅", "요약", "SOAP", "근거"])
    with tab_charting:
        if contents and '차팅' in contents:
            for content in contents['차팅'].split('\n'):
                st.write(content)
        else:
            st.text_area("-", height=300, key=f'charting_area_{key}')
    with tab_summary:
        if contents and '요약' in contents:
            for content in contents['요약'].split('\n'):
                st.write(content)
        else:
            st.text_area("-", height=300, key=f'charting_summary_{key}')
    with tab_soap:
        if contents and 'SOAP' in contents:
            for content in contents['SOAP'].split('\n'):
                st.write(content)
        else:
            st.text_area("-", height=300, key=f'charting_soap_{key}')
    with tab_reason:
        if contents and '근거 정리' in contents:
            for content in contents['근거 정리'].split('\n'):
                st.write(content)
        else:
            st.text_area("-", height=300, key=f'charting_reason_{key}')


def main():
    # UI 레이아웃 설정
    upload_dir = 'upload'
    os.makedirs(os.path.join(get_datapath(), upload_dir), exist_ok=True)

    # 타이틀 설정
    st.title(f"한의원 상담 내용 관리 시스템(API Server Status : {':green_heart:' if call_status_api() else ':broken_heart:'})")

    with st.container():
        # 파일 업로드 섹션
        st.subheader('파일 업로드')
        uploaded_file = st.file_uploader("상담 파일 업로드", type=['m4a', 'mp3', 'wav'])

        # 업로드된 파일 정보를 보여주는 위젯
        if uploaded_file is not None:
            st.text(uploaded_file.name)
        else:
            st.text("여기에 파일 이름이 표시됩니다.")

        if uploaded_file is not None:
            # 오디오 파일 저장
            if uploaded_file.name.split('.')[-1].lower() in ['m4a', 'mp3', 'wav']:
                audio_bytes = uploaded_file.read()
                st.audio(audio_bytes, format=f"audio/{uploaded_file.name.split('.')[-1]}")
                # 오디오 파일 저장
                with open(os.path.join(get_datapath(), upload_dir, uploaded_file.name), "wb") as f:
                    f.write(audio_bytes)
                if st.session_state.filename != uploaded_file.name:
                    init_session()
                st.session_state.filename = uploaded_file.name
            else:
                st.write(f"지원하지 않는 파일 타입니다.  {uploaded_file.name.split('.')[-1].lower()}")

    with st.container():
        st.subheader('STT 변환')
        sts_method = 'api' if st.selectbox('모델 선택', ('API', 'Local')) == 'API' else 'local'
        if st.session_state.filename is not None:
            if st.button('TEXT 추출', type='primary'):
                # STT 변환
                with st.spinner('오디오로부터 텍스트 추출중입니다...'):
                    st.session_state.stt_result = call_sts_api(sts_method, upload_dir)
                    st.session_state.stt_method = sts_method
            if st.session_state.stt_result is not None:
                with open(os.path.join(get_datapath(), st.session_state.stt_result['text_path']), "r",
                          encoding='utf8') as f:
                    with st.expander('STT 결과', expanded=False):
                        st.write(f"걸린 시간 : {st.session_state.stt_result['progress_time']} s")
                        st.write(
                            f"추출 횟수 : {st.session_state.stt_result['retry']}, temperature : {st.session_state.stt_result['temperature']}")
                    with st.expander('상담 내용', expanded=True):
                        st.write(f.read())
            else:
                print('stt result None')
        else:
            st.button('STT 변환', type='primary', disabled=True)

    # 요약 및 기록 섹션
    with st.container():
        st.subheader('상담 요약 및 기록')
        left_col, right_col = st.columns(2)

        with left_col:
            creat_llm_result(key='left')

        with right_col:
            creat_llm_result(key='right')


if __name__ == '__main__':
    print('-' * 10, ' refresh ', '-'*10)
    st.set_page_config(page_title='한방오남매의하모니', layout="wide")
    with open('config/auth.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
    authenticator = Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )
    name, authentication_status, username = authenticator.login('main')
    if st.session_state["authentication_status"]:
        _, login_name, login_area = st.columns([8, 1, 1])

        with login_name:
            st.markdown(f"<div style='color: red;'>Welcome *{st.session_state['name']}*</div>", unsafe_allow_html=True)
        with login_area:
            authenticator.logout('logout')
        if 'filename' not in st.session_state:
            init_session()
        main()
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')
