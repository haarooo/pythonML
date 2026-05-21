# PythonML Practice7: 로지스틱 분류
# 데이터 출처: https://www.kaggle.com/code/anshigupta01/iris-flower-classification

# [단계 1] 데이터 로드 및 독립/종속 변수 추출
# 파일명: ./Iris.csv
# 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' 4개 열을 독립 변수 X로,
# 'Species' 열을 종속 변수 y로 추출하세요.

import pandas as pd
df = pd.read_csv('./day05/Iris.csv')
iris_input = df[['SepalLengthCm' , 'SepalWidthCm' , 'PetalLengthCm' , 'PetalWidthCm']]
iris_target = df['Species']

print(iris_input)
print(iris_target)


# [단계 2] 훈련용 / 테스트용 데이터 분리
from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split(iris_input , iris_target , test_size=0.2 , random_state=42)

# [단계 3] 데이터 표준화 (Standardization), 스케일러
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scale = ss.transform(train_input)
test_scale = ss.transform(test_input)

# [단계 4] 로지스틱 분류 모델 학습 (Logistic Regression)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=3 , max_iter=100)
lr.fit(train_scale , train_target)

# [단계 5] 모델 평가 및 분류 정확도(Accuracy) 확인 * 테스트 세트의 정확도가 0.95 이상이 나오도록 설정
print(lr.score(test_scale , test_target))

# [단계 6] 학습한 종속 변수 출력
print(lr.classes_)

# [단계 7] 테스트 세트의 앞선 5개 샘플 데이터에 대해 모델이 예측한 클래스를 출력하세요.
print(lr.predict(test_input[:5]))




# PythonML Practice9: 패션 구매 카테고리 예측 시스템 구축 ( SGD )
# 데이터 출처: https://drive.google.com/file/d/1B5pVnqexkvh4D2l5c542HFaKmPjT5Iab/view?usp=sharing
# [상세 요구사항]
# 1. 고객의 인적 정보 및 구매 패턴 데이터를 데이터베이스(DB)에 적재 및 관리한다.
# 2. 관리자가 Spring Boot REST API를 호출하여, 데이터베이스에 저장된 전체 데이터를 기반으로 예측 모델을 (재)학습하고 최신화할 수 있어야 한다.
# 3. 일반 사용자가 Spring Boot REST API를 호출하여 4개 핵심 변수(나이, 성별, 유입 채널, 선호 스타일)를 입력하여 요청하면, 모델이 예측한 구매 옷 카테고리(0~7)를 실시간으로 반환한다.
# # 학습된 예측 모델의 예측 정밀도를 보장하기 위해 평가지표인 정확도가 최소 90% 이상(0.90 이상)을 달성해야 한다.
# # 프론트엔드 대신 Talend API를 사용한다.
# # REST API는 Spring API 2개와 FastAPI 2개로 구성한다.
# # 관리자와 일반 사용자는 FastAPI에 직접 접근하지 않으며, 모든 요청은 Spring API를 통해 처리한다.
# # 2인 ( 1:스프링 2:파이썬 )   