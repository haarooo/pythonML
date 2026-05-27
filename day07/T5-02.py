# ==========================================
# 1. 데이터 준비 
# ==========================================

import pandas as pd
text_data = [
    "사과 바나나 과일 식사",
    "포도 멜론 단맛 과일",
    "삼성 스마트폰 아이폰 출시 갤럭시 전자기기", 
    "컴퓨터 모니터 마우스 키보드 전자기기 제품", 
    "과일 딸기 주스 음료 디저트",
    "전자기기 노트북 인공지능 그래픽카드 컴퓨터 성능"
]
df = pd.DataFrame({'자연어_문장': text_data})

# 문자열에 대한 군집은 범주형이라서 힘들다 , kmodels 모델 사용하거나 딥러닝 
# 범주형 --> 수치형(백터화) 변경 , 텍스트(자연어) 수치형 백터로 변환
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer()
vector = tv.fit_transform(df['자연어_문장'])

# 백터화 스케일링 , 각 문장별 백터 최대 길이를 1로 고정
from sklearn.preprocessing import normalize
vector = normalize(vector)
print(vector)

# k-means 군집 모델
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2 , random_state=42)
km.fit(vector)
df['예측군집'] = km.labels_
print(df)


# 백터 구성 예시
# 단순 빈도 계산
# 2개의 문장이 존재 "사과 바나나 과일 식사", "포도 멜론 단맛 과일"
# 전체 문장 나열하여 총 7개 키워드/단어 추출 사과,바나나,과일,식사,포도,멜론,단맛
# 첫번째 문장은 7가지 키워드에 존재하는 검토 , 존재하면1 , 존재하지 않음 0
    # 첫번째 문장 = 사과,바나나,과일,식사,포도,멜론,단맛 -> 1,1,1,1,0,0,0
    # 두번째 문장 = 사과,바나나,과일,식사,포도,멜론,단맛 -> 0,0,1,0,1,1,1

# 스케일링
# 1,1,1,1,0,0,0 제곱의 합으로 총합을 1로 만든다
# 
