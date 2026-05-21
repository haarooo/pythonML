
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

# 확률적인 경사하강법
# fit 모델학습 에서는 정답도 같이 학습중이다 , 예측(y)값 과 실제 정답간의 오차 측정
# 경사하강법 (수많은 경우의수 계산하여 판단 , 컴퓨터가 좋아야한다)
# 경사하강법 vs 확률 경사하강법(SGD정확도 낮지만 학습속도가 빠르다 : 미니배치)

# log_loss
# 로그 로스 함수는 0과 1의 확률 값이 아닌 오차 값을 측정

# 분류 모델
from sklearn.linear_model import SGDClassifier
# random_state : SGD가 전체 데이터 학습이 아닌 일부 자료 가지고 학습하는데 사용 되는 분리 기준
# max_iter : (반복)계산 횟수 , 미니 배치이므로 전체 데이터셋을 10 이면 10반복 학습하여 모델 성공 향상<과적합> / 최적의 정확도에서 멈춤(에포크)
# tol=None : 최적의 정확도를 찾아도 계속 반복학습 설정
sc = SGDClassifier(loss='log_loss' , random_state=42 , max_iter=10 , tol=None)
sc.fit(train_scale , train_target)
print(sc.score(test_scale , test_target))
print(sc.predict(test_scale[:3]))

# 
sc.partial_fit(train_scale , train_target)
print(sc.score(test_scale , test_target))

# 최적의 학습횟수(에포크)
sc = SGDClassifier(loss='log_loss' , random_state=42)
train_score = []
test_score = []


import numpy as np
classes = np.unique(train_target) # 정답지의 중복제거한 고유 정답만 추출

for i in range(0 ,300) :
    sc.partial_fit(train_scale , train_target , classes=classes)

    train_score.append(sc.score(train_scale , train_target))
    test_score.append(sc.score(test_scale , test_target))



# 정확도 시각화 , 과대적합인지 과소적합인지 판단
import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.show()


# alpha 0.001 : 힌지 함수는 경계면(애매/아슬)에 있는 자료들을 찾는 기준
sc = SGDClassifier(loss='hinge' , max_iter=100 , random_state=42 , alpha=0.001)
sc.fit(train_scale , train_target)

print(sc.score(train_scale , train_target))
print(sc.score(test_scale , test_target))

# 로지스틱 회귀 : 확률 이용한 분류
# SGD(확률 경가하강법) : loss='log_loss' vs loss= 'hinge'

# loss='log_loss'
    # 도미일 확률이 50%기울기 절편으로 수없이 조정하여 확률 100%만드는 방법
    
# loss='hinge'
    # 도미 확률이 50% , 0인 지점인 애매/아슬한 (경계선) 자료만 가지고 확률 조정하는 방법