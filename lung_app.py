import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 현재 폴더의 폰트 사용 (GitHub 저장소에 NanumGothic.ttf가 있어야 합니다)
font_path = "./NanumGothic.ttf"
if os.path.exists(font_path):
    fontprop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = fontprop.get_name()
else:
    fontprop = None

plt.rcParams['axes.unicode_minus'] = False

# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="폐암 환자 군집 분석",
    page_icon="🫁",
    layout="centered"
)

# =========================
# 모델 & 스케일러 불러오기 (안전 구조 적용)
# =========================
@st.cache_resource
def load_machine_learning_assets():
    try:
        model = joblib.load("lung_model.pkl")
    except FileNotFoundError:
        st.error("🚨 'lung_model.pkl' 파일을 찾을 수 없습니다. GitHub 업로드 상태를 확인해 주세요.")
        st.stop()
        
    try:
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        st.error("🚨 'scaler.pkl' 파일을 찾을 수 없습니다. GitHub 업로드 상태를 확인해 주세요.")
        st.stop()
        
    try:
        df = pd.read_csv("lung.csv")
    except FileNotFoundError:
        df = None
        st.sidebar.warning("⚠️ 'lung.csv' 파일이 없어 배경 데이터 시각화가 제한됩니다.")
        
    return model, scaler, df

model, scaler, df = load_machine_learning_assets()

# 배경 데이터 군집 미리 예측 (데이터가 존재할 때만)
if df is not None:
    # 원본 CSV 컬럼 유연하게 체크 및 한글화
    age_col = 'Age' if 'Age' in df.columns else ('age' if 'age' in df.columns else '나이')
    smoke_col = 'Smokes' if 'Smokes' in df.columns else ('smokes' if 'smokes' in df.columns else ('Smoking' if 'Smoking' in df.columns else '흡연'))
    alkhol_col = 'Alkhol' if 'Alkhol' in df.columns else ('alkhol' if 'alkhol' in df.columns else ('Alcohol' if 'Alcohol' in df.columns else '음주'))
    
    # 💡 [해결 핵심 1] 모델 학습 당시와 글자 하나까지 똑같이 '나이', '흡연', '음주'로 명명합니다.
    X_orig = df[[age_col, smoke_col, alkhol_col]]
    X_orig.columns = ['나이', '흡연', '음주']
    
    # 스케일러 예측 진행
    X_scaled = scaler.transform(X_orig)
    df['cluster'] = model.predict(X_scaled)
    
    # 시각화용 데이터프레임 이름 통합
    df['나이'] = X_orig['나이']
    df['흡연'] = X_orig['흡연']
    df['음주'] = X_orig['음주']

# =========================
# 제목
# =========================
st.title("🫁 폐암 환자 군집 분석 시스템")

st.markdown("""
AI가 환자의 특성을 분석하여  
어떤 군집(유형)에 속하는지 예측합니다.
""")

st.divider()

# =========================
# 사용자 입력 (화면에는 이쁘게 양 표기)
# =========================
st.subheader("📋 환자 정보 입력")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("나이", min_value=0.0, max_value=120.0, value=50.0)

with col2:
    smoking = st.number_input("흡연량", min_value=0.0, value=10.0)

with col3:
    alcohol = st.number_input("음주량", min_value=0.0, value=5.0)

st.divider()

# =========================
# 예측 버튼 및 결과 시각화
# =========================
if st.button("🔍 군집 분석하기", use_container_width=True):

    # 💡 [해결 핵심 2] 스케일러가 애타게 요구한 학습 전용 피처명 ['나이', '흡연', '음주'] 그대로 매핑합니다.
    new_patient = pd.DataFrame([[age, smoking, alcohol]], columns=['나이', '흡연', '음주'])

    # 스케일링 및 군집 예측
    new_patient_scaled = scaler.transform(new_patient)
    pred_cluster = model.predict(new_patient_scaled)
    cluster_num = pred_cluster[0]

    # 결과 출력
    st.success(f"이 환자는 {cluster_num}번 군집에 속합니다.")
    st.write("0번은 매우 건강군, 1번은 건강군, 2번은 정상군, 3번은 위험군입니다")

    # =========================
    # 시각화 (한글 깨짐 방지 폰트 주입)
    # =========================
    fig, ax = plt.subplots(figsize=(8, 6))

    if df is not None:
        # 기존 데이터 산점도 분포 배경 생성
        scatter = ax.scatter(df['흡연'], df['음주'], c=df['cluster'], alpha=0.5, cmap='viridis', zorder=2)
        
        # 범례 타이틀 및 요소 한글화 적용
        legend1 = ax.legend(*scatter.legend_elements(), title="군집 구분", loc="upper right")
        if fontprop:
            plt.setp(legend1.get_title(), fontproperties=fontprop) 
            for text in legend1.get_texts():
                text.set_fontproperties(fontprop)
        ax.add_artist(legend1)

    # 사용자 위치 표시 (별표 마커)
    ax.scatter(smoking, alcohol, color='black', s=300, marker='*', label='현재 환자 위치', zorder=3)

    # 축 이름 및 타이틀 설정
    if fontprop:
        ax.set_xlabel("흡연량", fontproperties=fontprop, fontsize=12)
        ax.set_ylabel("음주량", fontproperties=fontprop, fontsize=12)
        ax.set_title("환자 군집 분포 시각화", fontproperties=fontprop, fontsize=14, pad=15)
        
        legend2 = ax.legend(loc="upper left")
        for text in legend2.get_texts():
            text.set_fontproperties(fontprop)
    else:
        ax.set_xlabel("흡연량", fontsize=12)
        ax.set_ylabel("음주량", fontsize=12)
        ax.set_title("환자 군집 분포 시각화", fontsize=14, pad=15)
        ax.legend(loc="upper left")

    ax.grid(True, linestyle='--', alpha=0.5, zorder=1)

    # Streamlit에 그래프 렌더링
    st.pyplot(fig)
