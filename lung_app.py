import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 현재 폴더의 폰트 사용
font_path = "./NanumGothic.ttf"

fontprop = fm.FontProperties(fname=font_path)

plt.rcParams['font.family'] = fontprop.get_name()
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
# 모델 & 스케일러 불러오기
# =========================
model = joblib.load("lung_model.pkl")
scaler = joblib.load("lung_scaler.pkl")

# =========================
# 학습 데이터 불러오기
# (시각화용)
# =========================
df = pd.read_csv("lung.csv")

X = df[['나이', '흡연량', '음주량']]

# 스케일링
X_scaled = scaler.transform(X)

# 기존 데이터 군집 예측
df['cluster'] = model.predict(X_scaled)

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
# 사용자 입력
# =========================
st.subheader("📋 환자 정보 입력")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "나이",
        min_value=0.0,
        max_value=120.0,
        value=50.0
    )

with col2:
    smoking = st.number_input(
        "흡연량",
        min_value=0.0,
        value=10.0
    )

with col3:
    alcohol = st.number_input(
        "음주량",
        min_value=0.0,
        value=5.0
    )

st.divider()

# =========================
# 예측 버튼
# =========================
if st.button("🔍 군집 분석하기", use_container_width=True):

    # 새로운 환자 데이터
    new_patient = pd.DataFrame(
        [[age, smoking, alcohol]],
        columns=['나이', '흡연량', '음주량']
    )

    # 스케일링
    new_patient_scaled = scaler.transform(new_patient)

    # 군집 예측
    pred_cluster = model.predict(new_patient_scaled)

    cluster_num = pred_cluster[0]

    # 결과 출력
    st.success(f"이 환자는 {cluster_num}번 군집에 속합니다.")
    st.write("0번은 매우 건강군, 1번은 위험군, 2번은 건강군입니다.")

    # =========================
    # 시각화
    # =========================

    fig, ax = plt.subplots(figsize=(8,6))

    # 기존 데이터 산점도
    scatter = ax.scatter(
        df['흡연량'],
        df['음주량'],
        c=df['cluster']
    )

    # 사용자 위치 표시
    ax.scatter(
        smoking,
        alcohol,
        s=300,
        marker='*'
    )

    ax.set_xlabel("흡연량", fontproperties=fontprop)
    ax.set_ylabel("음주량", fontproperties=fontprop)
    ax.set_title("군집 시각화", fontproperties=fontprop)

    st.pyplot(fig)
