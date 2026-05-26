import pandas as pd

df = pd.read_csv('./day06/wine.csv')
data = df[['alcohol' , 'sugar' , 'pH']]
target = df['class']

from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split(data , target ,  random_state=42)

# 트리의 앙상블 : 학습한 모델에서 오답들을 서로 상괘하고 정답을 강화하여 예측정확도를 높이는 방법 # 여러가지 방법 존재
# 랜덤포레스트
# 모든 특성 사용
#   부트스트랩 샘플링 : 전체 훈련데이터 중에서 무작위로 샘플 선정
#   무작위 특성 : 전체 특성 중에서 무작위로 샘플 선정한다
# 즉 모든 특성들을 사용하여 다양한 

# oob(out - of bag)무작위(중복허용) 선정시 1번도 선정 안된 자료들을 평가용으로 사용
# 1 ,2 ,3 ,4 ,5 중에서  무작위로 1,2,3,5,5 선정하면 선정 안된 4는 샘플
# 4(샘플) 가지고 학습으로 검증한다 ==> oob_score , 자체 검증
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(oob_score=True , n_jobs=-1 , random_state=42)

# 교차 검증
from sklearn.model_selection import cross_validate
score = cross_validate(rf , train_input , train_target , n_jobs=-1)
print(score)

import numpy as np
print(np.mean(score['test_score']))

# 특성 중요도
rf.fit(train_input , train_target)
print(rf.feature_importances_) # 결정트리 보다 조금 더 골고루 분산되었다
# 분류 모델중에서는 (간단한 모델)로지스틱회귀모델 vs (복잡한 모델)트리모델(+앙상블)

# 엑스트라 트리
#   모든 트리가 전체 샘플 자료를 학습한다
#   무작위 노드 분할 : 예 sugar 특성을 무작위로 1.4 기준으로 잘라서 분리한다 ,무작위라서 오답이 많이 발생
# 예시 나이 특성에 20~60세가 존재한 경우 노드분할 예시
#   Tree(노드) 에서 무작위로 나이 특성을 29세 이상 조건을 만든다.(수학적인 계산이 업어서 빠르다)
#   Tree에서 무작위로 나이 특성을 50세 이상 조건을 만든다
# 즉 노드마다 서로 다른 기준점을 분할하여 다양성 확보한다 , 게산식이 없어서 허술한 방법이지만 학습 수와 방대한 양으로 오차 극복
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs=-1 , random_state=42) # 모델 생성
score = cross_validate(et , train_input , train_target , n_jobs=1)
print(score)
print(np.mean(score['test_score']))
et.fit(train_input , train_target)
print(et.feature_importances_)




# 그래디언트 부스팅
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=42)
score = cross_validate(gb , train_input , train_target , n_jobs=-1)
print(score)
print(np.mean(score['test_score']))

# 특성 중요도
gb.fit(train_input , train_target)
print(gb.feature_importances_)

# 히스토그램 기반 그래디언트 부스팅
# 히스토그램 : 연속적인 구간으로 나누어 표현
# 특성 정량화 : 연속적인 구간을 256개의 구간(정수)로 나누어서 단순화한다
# 분할 기준 : 자식 노드를 만들때 256개 구간 기준으로 분할한다

# 예 :  180 , 180.8 , 180.3 처럼 소수점 단위의 촘촘히 떨어져있는 데이터
# 180~181까지 하나의 구간으로 묶어서 계산한다
# 미세한 소수점 오차는 과감하게 버린다 , 메모리 절약과 속도 향상
from sklearn.ensemble import HistGradientBoostingClassifier
hb = HistGradientBoostingClassifier(random_state=42)
score = cross_validate(hb , train_input , train_target , n_jobs=-1)
print(score)
print(np.mean(score['test_score']))


# 외부 라이브러리 앙상블 
# 1. 
# pip install xgboost
# 장점1 : 손실함수(라쏘,릿지) 규제 내장 , 과적합 방지
# 장점2 : 병렬처리로 CPU 캐시 사용(속도 향상)
from xgboost import XGBClassifier
xg = XGBClassifier(tree_method = 'hist' , random_state = 42)
score = cross_validate(xg , train_input , train_target , n_jobs=-1)
print(np.mean(score['test_score']))

# 2. pip install lightgbm , 분류모델에서 사용빈도 큼
# 장점1 : 그래디언트 부스팅 기반으로 부모가 자식에게 오차를 물려주는 방법
# 장점2 : 부모(왼쪽노드 , 오른쪽노드) 기준으로 오차가 큰 노드 부터 처리하는 방법(비대칭 구조)
# 예 작은 오차 노드는 무시하고 큰 오차 노드부터 최적화, 과대적합 위험있다. 최소 노드 샘플로 과대적합 방지
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(random_state=42)
score = cross_validate(lgb , train_input , train_target , n_jobs=-1)
print(np.mean(score['test_score']))

# 3. pip install catboost
# 장점1 : 문자형으로 된 자료들을 숫자로 변경할 때 (category)  
# 장점2 : 문자형을 숫자형으로 변경하는 시뮬레이션 예측에 대한 시뮬레이션 수치화 한다.
# 예 자식 노드가 동일한 조건으로 분리하고 자식노드에 대한 샘플 자료를 수치화 하여 예측 속도 향상 한다.
from catboost import CatBoostClassifier
cat = CatBoostClassifier(random_state=42 , verbose=0)
scroe = cross_validate(cat , train_input , train_target , n_jobs=-1)
print(np.mean(score['test_score']))


# 분류모델 선정 

# 앙상블(앞전 계산에 사용도니 오차/결과를 다음/전체에 정확도 향상하는데 상쇄 방법)
# 1. 랜덤포레스트 : 샘플/특성 무작위로 선정하여 모델 학습 , 튜닝 시간이 부족하거나 베이스 모델 사용 
# 2. 엑스트라트리 : 노드분할기준을 무작위로 선정하여 모델 학습 , 성능 변동이 있더라도 학습 속도 개선 사용
# 3. 그레디언트부스팅 : 부모노드에서 오차를 자식노드에게 전달하는 모델 학습 , 학습 속도 보다 정교한 모델 사용
# 4. 히스토그램 기반 그래디언트 부스팅 : 연석된 샘플들을 구간(256개)을 만들어서 모델 학습 , 전처리 시간이 부족하거나 학습 속도 개선 사용 
# 5. xgboost : 손실함수(라쏘,릿지) 규제 사용과 CPU 캐시 사용하는 모델 학습
# 6. *lightgbm : 오차가 큰 노드 부터 최적화하는 모델 학습 , 학습 속도 향상 모델 사용 
# 7. catboost : 데이터가 문자형으로 대다수인 경우와 튜닝을 최소화하는 모델 학습