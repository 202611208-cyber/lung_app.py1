import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. 모델, 스케일러, 데이터셋 로드
@st.cache_resource
def load_assets():
    model = joblib.load('lung_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    try:
        df = pd.read_csv('lung.csv')
        
        # 원본 CSV가 영문일 경우 주피터 노트북처럼 한글로 강제 변환
        rename_dict = {
            'Age': '나이', 'age': '나이',
            'Smokes': '흡연', 'smokes': '흡연',
            'Alkhol': '음주', 'alkhol': '음주',
            'Result': '폐암여부', 'result': '폐암여부'
        }
        df = df.rename(columns=rename_dict)
        
        # 원본 CSV에 'cluster' 열이 없을 수 있으므로 직접 채워줌
        if 'cluster' not in df.columns:
            features = df[['나이', '흡연', '음주']]
            features_scaled = scaler.transform(features)
            df['cluster'] = model.predict(features_scaled)
            
    except FileNotFoundError:
        df = None
    return model, scaler, df

model, scaler, df = load_assets()

# 2. 웹앱 타이틀 및 레이아웃 설정
st.title("🫁 폐암 위험도 기반 환자 군집 예측 시스템")
st.markdown("환자의 나이, 흡연량, 음주량을 입력하여 어떤 건강 군집에 속하는지 확인하고 시각화합니다.")
st.divider()

# 3. 사이드바 - 사용자 데이터 입력 받기
st.sidebar.header("📝 환자 데이터 입력")

age = st.sidebar.number_input("나이 입력", min_value=0, max_value=120, value=35, step=1)
smokes = st.sidebar.number_input("흡연 입력", min_value=0.0, max_value=100.0, value=1.0, step=0.5)
alkhol = st.sidebar.number_input("음주 입력", min_value=0.0, max_value=100.0, value=1.0, step=0.5)

# 4. 입력 데이터 데이터프레임 변환 및 예측 변환
new_patient = pd.DataFrame([[age, smokes, alkhol]], columns=['나이', '흡연', '음주'])
new_patient_scaled = scaler.transform(new_patient)
pred_cluster = model.predict(new_patient_scaled)[0]

# 5. 결과 출력
st.subheader("📊 예측 결과")

cluster_info = {
    0: "🟢 0번 군집 (매우 건강군)",
    1: "🟡 1번 군집 (건강군)",
    2: "🟠 2번 군집 (중간 그룹)",
    3: "🔴 3번 군집 (강한 폐암 위험군)"
}
cluster_desc = cluster_info.get(pred_cluster, f"{pred_cluster}번 군집")

st.markdown(f"### 이 환자는 **{cluster_desc}**에 속합니다.")

# 6. 결과 시각화
st.subheader("📍 군집 내 환자 위치 시각화")

fig, ax = plt.subplots(figsize=(8, 6))

# 💡 [팁] Streamlit Cloud(리눅스) 환경에서는 맑은고딕이 없어 한글이 깨질 수 있으므로
# 축 이름은 영어로 지정하여 배포 에러 및 글꼴 깨짐을 안전하게 방지합니다.
plt.rcParams['axes.unicode_minus'] = False

# 배경에 기존 데이터들 뿌려주기
if df is not None:
    scatter = ax.scatter(df['흡연'], df['음주'], c=df['cluster'], alpha=0.5, cmap='viridis', zorder=2)
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper right")
    ax.add_artist(legend1)

# 새 환자 위치 표시
ax.scatter(smokes, alkhol, color='black', s=300, marker='X', label='New Patient', zorder=3)

# 웹 서버 환경 호환을 위해 영문 라벨 권장
ax.set_xlabel('Smoking')
ax.set_ylabel('Drinking')
ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
ax.legend(loc="upper left")

# Streamlit 화면에 그래프 출력
st.pyplot(fig)
