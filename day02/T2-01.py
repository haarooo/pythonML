
import pandas as pd
df = pd.read_csv('./machine/Fish.csv')

# Perch만 추출
target_fish = df[df['Species'].isin(['Perch'])]
print(target_fish)
target_fish.info()

perch_length = target_fish['Length2'].values
perch_weight = target_fish['Weight'].values
print(perch_length , perch_weight)

# 길이에 따른 무게 예측
import matplotlib.pyplot as plt
plt.scatter(perch_length , perch_weight)
plt.show()

# 학습 모델 만들기 , 학습용과 테스트용 분리 = 모델평가에 사용
from sklearn.model_selection import train_test_split
# random_state = 분리할 때 사용되는 난수값 , 난수값에 따라 분리한다. # 고정값 넣어주면 항상 동일한 분리 값 넣을 수 있다 # 0~32 사이
train_input, test_input , train_target , test_target = train_test_split(perch_length , perch_weight , test_size=0.3 , random_state=42)

# 자료형식 구성 
import numpy as np
array = np.array([1,2,3,4])
print(array.shape) # shape 배열의 모양 반환
array2 = np.array([[1,2,] , [3,4]])
print(array2.shape)
print(train_input.shape) # 사이킷런 모델들은 1차원배열 학습이 불가능
print(train_input)

train_input = train_input.reshape(-1 , 1)
train_target = train_target.reshape(-1 ,1)
test_input = test_input.reshape(-1,1)
# test_target = test_target.reshape(-1,1)
print(train_input)

# 모델 학습
from sklearn.neighbors import KNeighborsClassifier # 최근접이웃 찾기 
from sklearn.neighbors import KNeighborsRegressor # 최근접이웃 회귀 

knr = KNeighborsRegressor()
knr.fit(train_input , train_target) # 모델 학습
print(knr.score(test_input , test_target))
print(knr.predict(test_input))
print(test_input) 

