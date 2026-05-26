import pandas as pd

df = pd.read_csv('./day06/wine.csv')
data = df[['alcohol' , 'sugar' , 'pH']]
target = df['class']

from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split(data , target , test_size=0.2 , random_state=42)

# 결정 트리 (분류)
from sklearn.tree import DecisionTreeClassifier # 의사결정 트리 분류
dt = DecisionTreeClassifier()
dt.fit(train_input,train_target)
# 모델 정확도 확인
print(dt.score(train_input ,train_target))
print(dt.score(test_input , test_target))

# 모델 예측
print(dt.predict(test_input[:5])) # 5개만 예측

# 결정 트리 시각화
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
# filled=True : 노드 색깔 다르게 표현
plot_tree(dt  , max_depth=1 , feature_names=['alcohoa' , 'sugar' , 'pH'] , class_names=['red' , 'white'] , filled=True) # max_depth = 기지수
plt.show()

# 트리 : 전체적인 구조 
# 노드 : 사각형 상자 하나 , 가장 위에 있는 노드를 루트(root)노드 
# 노드 속성 : 
    # value=[예측타겟수] # [85,2057] 0으로 예측하는 수가 85개 , 1로 예측하는 수가 2057개 뜻
    # gini = 불순도 # 0.075 # 0과 가까울수록 순수하다 # 0.5에 가까울수록 혼란하다
    # sugar 특성 # sugar <= 4.05 보다 작으면 true(왼쪽 노드로 이동) , false(오른쪽 노드로 이동)
    

# 특성 중요도 
# 각 특성이 트리 모델레 얼마나 중요한 역할 하는지 수치
print(dt.feature_importances_) # 각 특성이 트리 모델에 얼마나 중요한 역할 하는지 수치  
print(dt.feature_importances_[0])

# 최소한의 불순도(gini) 설정 , 최적의 파라미터
dt = DecisionTreeClassifier(random_state=42 , min_impurity_decrease= 0.0005)
dt.fit(train_input , train_target)
print(dt.score(train_input , train_target)) 
print(dt.score(test_input , test_target)) # 과대적합 최소화
