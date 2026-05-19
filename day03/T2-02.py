
# 모델 : 데이터를 학습하는 프로그램/라이브러리
    # K-NN : 가까운 이웃 기준의 예측
        # 하이퍼파라미터
# 학습 : 데이터의 규칙 찾는 과정
# 예측 : 학습된 모델로 새로운 데이터 추론 과정
# 특성 : 학습에 입력되는 정보
# 타깃 : 학습에 정답되는 정보 
# 표준화(스케일링) : 0~1 사이로 크기 맞춤
# 과소적합 : 너무 단순한 경우
# 과대적합 : 너무 암기된 경우 

import pandas as pd
df = pd.read_csv('./day03/Fish.csv')
fish_data = df[df['Species'].isin(['Perch'])]
perch_length = fish_data['Length2'].values
perch_weight = fish_data['Weight'].values

# 학습 자료와 테스트 자료 분리
from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split(perch_length , perch_weight , test_size=0.3 , random_state=42)

# 학습 하기 전에 사이킷런 모델들을 2차원 배열화
train_input = train_input.reshape(-1 ,1)
test_input = test_input.reshape(-1 , 1)

# 최근접 이웃 회귀 모델 훈련
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(train_input , train_target) # 모델 학습
print(knr.score(test_input , test_target))

# 임의의 값으로 예측하기
print(knr.predict([[50]]))
print(knr.predict([[100]]))

# 문제점 : k-최근접 이웃 의 문제점은 단순한 주변 이웃의 평균으로 예측하기 때문에 최댓값을 벗어나면 항상 동일한 값으로 예측한다.
# 즉] 소규모 또는 간단한 예측 프로그램 에서만 사용된다.

# 다른 모델 사용하기
from sklearn.linear_model import LinearRegression # 선형회귀 모델
lr = LinearRegression()
lr.fit(train_input , train_target)
print(lr.predict([[50]]))
print(lr.predict([[100]]))

# 직선공식(1차방정식) : y(예측값) = ax(특성)+b
print(lr.coef_) # 기울기밧 반환 , 특성의 가중치 
print(lr.intercept_) # y절편 반환 , 편향 x가 0일때 y의 값
# x와 y가 직선 관계이며 실 자료들은 물고기 길이가 1씩 증가할 때 무게가 꼭 비례 증가 하지 않는다 
# 초반에는 길이에 따라 무게가 3배 증가하다가 중/후반에는 무게가 점차 줄어들 수 있다 

# 시각화 
import matplotlib.pyplot as plt
plt.scatter(train_input , train_target) # 단순 선형 평가
plt.scatter(50 , 1238)
plt.scatter(100 , 3101)
plt.plot([15 , 100] , lr.predict([[15] , [100]])) # 회귀선 그리기 
plt.show()
print(lr.score(test_input , test_target))


# 다항 선형회귀 모델 # 2차 방정식
# 직선공식(1차 방정식) : y(예측) = w(가중치) x X(특성) + B(절편)
# 곡선공식(2차 방정식) : Y(예측) = W(가중치) x X(특성)제곱 + W(가중치) X(특성)제곱 + B(절편) 
# x(특성) 제곱 : 물고기 길이에 제곱 , 최적의 제곱 찾아서 정확도 최적화 한다
import numpy as np
train_poly = np.column_stack((train_input**2 , train_input))

lr = LinearRegression()
lr.fit(train_poly , train_target)

# 예측할 자료 , 길이 : 50인 무게 예측
print(lr.predict([[50**2 , 50]]))

# 여러개 예측
point = np.arange(15 ,50)
print(point)
point_poly = np.column_stack([point**2 , point])

plt.scatter(train_input , train_target)
plt.plot(point , lr.predict(point_poly))
plt.show()
test_poly = np.column_stack([test_input**2 , test_input])
print(lr.score(test_poly , test_target))

