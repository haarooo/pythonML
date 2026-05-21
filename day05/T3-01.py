
import pandas as pd
df = pd.read_csv('./day05/Fish.csv')

# Species : 어종 7개
# 특성 : Weight , Length1 , Length2 , Length3 , Height , Width

fish_input = df[['Weight' , 'Length1' , 'Length2' , 'Length3' , 'Height' , 'Width']] 
fish_target = df['Species']

from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split(fish_input , fish_target , test_size=0.25 , random_state=42)

# 스케일링
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scale = ss.transform(train_input)
test_scale = ss.transform(test_input)

# 이진 분류 , 로지스틱 회귀 모델 = 이진분류 = 시그모이드 함수(공식)
# 선형 방정식의 출력값을 0과 1(확률/분류) 사이의 값으로 변환해주는 공식/함수
import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-100 , 100 , 0.1) # -5 부터 5까지 0.1씩 증가하는 리스트
ph1 = 1/(1+np.exp(-z)) # 시그모이드 공식
plt.plot(z , ph1) # 시그모이드 시각화
# plt.show()

indexs =  (train_target == 'Bream' ) | (train_target == 'Smelt')
train_bream_smelt = train_scale[indexs]
target_bream_smelt = train_target[indexs]
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(train_bream_smelt , target_bream_smelt) # 도미와 빙어만 학습

# 이진분류 모델 예측
print(lr.predict(train_bream_smelt[: 3])) # 3개만 예측
print(lr.predict_proba(train_bream_smelt[:3])) # 3개만 예측 확률

# 임계값은  0.5 기준으로 0.5 이상이면
# [[0.95684973 0.04315027]
#  [0.99825845 0.00174155]
#  [0.02395382 0.97604618]]


# 다중 분류 # 로지스틱 회귀
# 하이퍼파라미터
# c = 규제를 완하하여 릿지/라쏘 모델 처럼 정확도 설정 가능
# max_iter : 다중분류 계산 횟수 , 생력시 기본값 100으로 최적의 정확도를 찾을 때 까지 계산 반복횟수 조정
lr = LogisticRegression(C=3 , max_iter=100)
lr.fit(train_scale , train_target) # 모든 어종학습
print(lr.predict(test_scale[:3]))
print(lr.predict_proba(test_scale[:3]))

# 모델 평가
print(lr.score(test_scale , test_target))


# 소프트맥스
from scipy.special import softmax
decision = lr.decision_function(test_scale[:3])
print(softmax(decision))
print(np.round(softmax(decision) , decimals=3))
print(lr.classes_) # 종속변수들 출력