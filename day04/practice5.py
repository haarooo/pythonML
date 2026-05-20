# PythonML Practice 5: 다항 규제 회귀 기반 성적 예측
# 데이터 출처: https://www.kaggle.com/datasets/shambhurajejagadale/student-performance-prediction-dataset

# [1] 데이터 분할: 범주형 변수를 제외한 6개 특성을 독립변수로, `exam_score`를 타깃으로 설정하고 8:2 비율 로 학습 및 검증 세트를 분리하시오.

import pandas as pd
df = pd.read_csv('./day04/student_dataset_10000_rows.csv')
exam_cause = df[['study_hours' , 'attendance' , 'sleep_hours' , 'internet_usage' , 'assignments_completed' , 'previous_score']].values
exam_score = df['exam_score'].values
print(exam_cause)

from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split(exam_cause , exam_score , test_size=0.2 , random_state=42)

# [2] 모델 전수 탐색: `LinearRegression`, `Ridge`, `Lasso` 모델과 다항 확장, 다양한 규제 강도 조합을 모두 학습시키시오.

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# 기본 선형 모델 학습 및 평가
lr.fit(train_input , train_target)
print(lr.score(train_input , train_target))

# 다항 선형 학습 및 평가
import numpy as np
train_poly = np.column_stack((train_input**2 , train_input))
test_poly = np.column_stack((test_input**2 , test_input))
lr = LinearRegression()
lr.fit(train_poly , train_target)
print(lr.score(test_poly , test_target))

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

optimi = []

for degree in [1,2,3,4,5]:
    poly = PolynomialFeatures(degree= degree ,include_bias=False)
    poly.fit(train_input)
    train_poly = poly.transform(train_input)
    test_poly = poly.transform(test_input)
    lr=LinearRegression()
    lr.fit(train_poly, train_target)
    r2 = lr.score(test_poly , test_target)
    # print(lr.score(test_poly , test_target))
    optimi.append({
        'r2' : r2 , 'model' : lr , 
        'poly' : poly , 'degree' : degree , 
        'scale' : None , 'alpha' : None
        })

    # 스케일링
    ss = StandardScaler()
    ss.fit(train_poly)
    train_scale = ss.transform(train_poly)
    test_scale = ss.transform(test_poly)
    # 규제 강도 , 릿지vs라쏘
    for alpha in [0.01 , 0.1 , 1, 10 ,100]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(train_scale , train_target)
        r2 = ridge.score(test_scale , test_target)
        # print("-----------------",alpha)
        # print(ridge.score(test_scale , test_target))

        optimi.append({
        'r2' : r2 , 'model' : ridge , 
        'poly' : poly , 'degree' : degree , 
        'scale' : ss , 'alpha' : alpha
        })
        
        lasso = Lasso(alpha=alpha)
        lasso.fit(train_scale , train_target)
        r2 = lasso.score(test_scale , test_target)
        # print("-----------------",alpha)
        # print(lasso.score(test_scale , test_target))

        optimi.append({
        'r2' : r2 , 'model' : lasso , 
        'poly' : poly , 'degree' : degree , 
        'scale' : ss , 'alpha' : alpha
        })


# [3] 최적 모델 선정: 테스트 데이터셋(`X_test`) 기준 최고의 결정계수를 달성하는 최적의 알고리즘, 차수, 알파 값을 자동 도출하고 추론 엔진에 매핑하시오.
# 최적 모델 찾기위한 리스트에서 결정계수(r2)가 가장 큰 모델 찾기
list = [{"data1" : 10 , "data2" :20} , {'data1' : 35 , "data2" : 15}]
print(max(list , key=lambda x : x['data1']))

best_optimi = max(optimi , key=lambda x:x['r2'])
best_model = best_optimi['model']
best_poly = best_optimi['poly']
scale = best_optimi['scale']
print(f'최적의 모델 : {best_model} , 다항특성 : {best_poly}, 스케일링 : {scale}')

        
# [4] 추론 함수 구현: 새로운 학생의 6가지 특성 데이터를 인자로 받아 최적 모델의 다항 구조와 스케일링 기준을 거쳐 성적을 예측하는 함수를 구현하시오.
# [5] 샘플 데이터 검증: 구현된 함수에 두 가지 대조군 샘플을 대입하여 시험성적을 예측하시오.
    # study_hours=9, attendance=95, sleep_hours=7, internet_usage=2, assignments_completed=18, previous_score=85
    # study_hours=2, attendance=60, sleep_hours=5, internet_usage=9, assignments_completed=4, previous_score=50

def exam_score(study_hours , attendance , sleep_hours , internet_usage , assignments_completed , previous_score) :

    list_data = [[study_hours , attendance , sleep_hours , internet_usage , assignments_completed , previous_score]]

    # 특성 공학 적용
    list_poly = best_poly.transform(list_data)

    # 스케일링 사용할지 여부 판단
    if scale is not None: # 만약에 scaler 사용한 모델이면
        list_poly = scale.transform(list_poly) # 스케일링 적용
    result = best_model.predict(list_poly)
    return result[0]


result = exam_score(9,95,7,2,18,85)
print(result)
result = exam_score(2, 60 , 5, 9 , 4, 50)
print(result)
