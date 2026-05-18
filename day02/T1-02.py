
import pandas as pd
df = pd.read_csv('./machine/Fish.csv')
df.info()

# 2. 필요한 어종 추출 : 조건식 대신 .isin()
target_fish = df[df['Species'].isin(['Bream' , 'Smelt'])]
print(target_fish)

# 필요한 특성 추출 : Length2 , weight
import numpy as np
# np.column_stack  두 리스트간에 동일한 요소로 2차원 리스트 구성
fish_data = np.column_stack((target_fish['Length2'] , target_fish['Weight']))
print(fish_data)

# 모델 학습을 하기 위한 정답지 , 도미 35마리  빙어 14마리
fish_target = np.concatenate( (np.ones(35) , np.zeros(14)))
print(fish_target)

# 학습 모델 만들기, 학습용 테스트용 분리 # 방대한 자료(억단위 이상)
from sklearn.model_selection import train_test_split
# 학습용 , 테스트용 , 학습용정답지 , 테스트용정답지 = train_test_split(학습자료 , 정답지 ,test_size = 테스트자료비율 )
train_input , test_input , train_target , test_target =train_test_split(fish_data , fish_target , test_size=0.3)  # 7:3비율로 분할
print(train_input.shape) # 49개 중에 학습용 7에 해당하는 자료가 34개
print(test_input.shape) # 49개 중에 테스트용3에 해당하는 개수가 15개

# 학습 모델 , K-최근접 이웃 분류기 모델
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier() # 모델 객체 생성
kn.fit(train_input , train_target) # 모델 학습
print(kn.score(test_input , test_target)) #모델 평가

# 임의의 값으로 학습 모델 예측하기
# 길이 : 25 , 무게 : 150의 물고기가 도미인지 빙어인지 예측하기
print(kn.predict([(25, 150)])) # 잘못된 예측

# 예측값 시각화
import matplotlib.pyplot as plt
print(train_input[:,0]) # 행슬라이싱 , 열슬라이싱 , 모든 행의 0번째 열만 추출 즉 길이만 추출
print(train_input[:,1]) # 행슬라이싱 , 열슬라이싱 , 모든 행의 1번째 열만 추출 즉 무제만 추출
plt.scatter(train_input[: , 0] , train_input[: , 1])
plt.scatter(25 , 150)
plt.show()

# 예측하기 위한 이웃들 확인
dist , indexs = kn.kneighbors([[25,150]])
plt.scatter(train_input[:, 0] , train_input[:,1])
plt.scatter(25, 100)
plt.scatter(train_input[indexs ,0] , train_input[indexs , 1])
plt.show()

# 표준화 필요성 : 공정하게 크기단위 맞추는 작업 > 길이와 무게 값의 차이가 커서 일관된 비교가 어렵다
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_input)
print(scaler.mean_)
print(scaler.scale_) #표준편차
train_scaled = scaler.transform(train_input) # 표준화(스케일링)

# 스케일링 시각화 , 모양의 차이는 없지만 단위가 표준화 되었다
plt.scatter(train_scaled[ : , 0] , train_scaled[: ,1])
plt.show()

# 스케일링 이후 재학습 모델 만들기
# 임의의 예측(스케일링)
new = scaler.transform([[25,150]])
print(kn.predict((new)))

# 예측에 사용된 이웃들 확인
dist , indexs = kn.kneighbors(new) 
kn.fit(train_scaled , train_target)
plt.scatter(train_scaled[ : , 0] , train_scaled[: ,1]) # 스케일링된 학습용
plt.scatter(new[: , 0] , new[: , 1]) # 스케일링된 예측값
plt.scatter(train_scaled[indexs , 0] , train_scaled[indexs , 1])
plt.show()