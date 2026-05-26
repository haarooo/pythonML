import pandas as pd

df = pd.read_csv('./day06/wine.csv')
data = df[['alcohol' , 'sugar' , 'pH']]
target = df['class']

from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split(data , target ,  random_state=42)

# 결정 트리
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_input , train_target)
print(dt.score(test_input , test_target))

# 교차검증
from sklearn.model_selection import cross_validate
# 교차검증은 전체 데이터를 N등분(폴드) 하며 돌아가면서 검증한다. , 기본값은 5등분
score = cross_validate(dt , train_input , train_target)
print(score)

import numpy as np
print(np.mean(score['test_score']))

from sklearn.model_selection import StratifiedKFold
splits = StratifiedKFold(n_splits=10 , shuffle=True , random_state=42)
score = cross_validate(dt , train_input , train_target , cv=splits)
print(np.mean(score['test_score']))


# 그리드 서치 , 최적의 파라미터 (변수/학습에 필요한 설정값) 찾기
from sklearn.model_selection import GridSearchCV
# 여러개 최소불순도 설정 
# 임의의 최소불순도 넣어서 리스트로 구성
params = {'min_impurity_decrease' : [0.0001 , 0.0002 , 0.0003 , 0.0004 , 0.0005]}

# 그리드서치 학습
# n_jobs= -1 : 컴퓨터에 모든 CPU 코어 사용하여 병렬연산
gs = GridSearchCV(DecisionTreeClassifier(random_state=42) , params , n_jobs=-1 )
gs.fit(train_input ,train_target)
dt = gs.best_estimator_ # 최적의 파라미터로 학습결과
print(dt.score(test_input , test_target)) # 
print(gs.best_score_)
print(gs.best_params_)
print(gs.cv_results_)

# 다중 파라미터 
params = {
    'min_impurity_decrease' : np.arange(0.0001 , 0.001 , 0.0001),
    'max_depth' : np.arange(5 , 20 , 1),
    # 노트 분할시 최저 샘플수 , 즉 최저 샘플수 보다 작으면 노드 분할 안함
    'min_samples_split' : np.arange(2 , 100 ,10),
    # 리프노드(마지막 노드) 최저 샘플수 , 즉 현재 리프노드가 최저 샘플수 보다 작으면 노드 분할 안함,
    'min_samples_leaf' : np.arange(1, 100 ,10)
}
# cv = 교차검증(n등분)
gs = GridSearchCV(DecisionTreeClassifier(random_state=42) , params , n_jobs=-1 , cv=5)
# 최저불순도(9가지) , 길이(15가지) = 최저분리샘플(10가지) , 최저리프샘플(10가지)
# 교차검증(N등분) 13000가지 조합 = 6만번의 학습 모델
gs.fit(train_input , train_target)
print(gs.best_params_)
print(gs.best_score_)

# 랜덤서치
# 조합 수가 많아지면 연산량이 많아져서 서버에 부하 발생할 수 있다.
# 랜덤서치란 고정된 값이 아니라 확률 분포 함수를 제공하여 무작위로 숫자를 뽑아 학습한다
from sklearn.model_selection import RandomizedSearchCV
# n_iter=100 정의된 조합수에서 무작위로 N개의 조합만 추출하여 학습한다
# 대략 13000 조합에서 100개만 무작위로 추출 # 교차검증 -> 500번 학습
rs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42) , params , n_iter=100 , n_jobs=-1 , cv=5 , random_state=42)
rs.fit(train_input , train_target)
print(rs.best_score_)
print(rs.best_params_)

