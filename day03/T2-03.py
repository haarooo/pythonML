
# 숭어의 길이 , 높이 , 두께(특성) 무게(타겟)
import pandas as pd
df = pd.read_csv('./day03/Fish.csv')
perch = df[df['Species'].isin(['Perch'])]
perch_full = perch[['Length2' , 'Height' , 'Width']].values
perch_weight = perch['Weight'].values

# 모델 검증
from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split(perch_full , perch_weight , test_size=0.2 , random_state=42)


# 다양한 특성을 추가로 만들어서 모델이 다양한 구조로 이해하기 위한 방법
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures()
poly.fit([[2]])
print(poly.transform([[2]]))
poly = PolynomialFeatures(include_bias=False)
poly.fit([[2,3]])
print(poly.transform([[2,3]]))

# 적용 3가지의 특성을 갖는다
poly = PolynomialFeatures(include_bias=False) # 다향특성 객체 생성
poly.fit(train_input) # 학습할 특성들은 대입
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly) # 3가지 특성

# 다형 회귀
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly , train_target)

print(lr.score(test_poly , test_target))
print(lr.score(train_poly , train_target))


# 스코어(점수) 결정계수
# 계수란 = 기울기와 가중치 즉 어떤 에측 결과에 얼마나 중요한 비중을 차지하는지
# 결정계수란? 0 ~ 1 사이의 값으로 예측한 값이 얼마나 잘 설명 하는지 나타내는 수치 
# 결정계수 계산식  , K-NN모델은 전체 계산식이 아닌 근접한 이웃 이용한 계산식 이므로 한계점
    # 타깃의 총 변동량 = SS_TOT = sum( (실제값 - 실제값평균)**2 )
    # 타깃의 오차 변동량 = SS_RES = sum( (실제값 - 예측값 )**2 )
    # 1(100%) - ( ss_res/ss_tot )


# 과대적합 확인
poly = PolynomialFeatures(degree=5 , include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)

# 모델학습
lr.fit(train_poly , train_target)

# 모델평가
# 과대적합 : 특정한 자료에만 과도한 학습을 통해 학습된것만 예측하고 세로운 자료에 대해서는 평가 불가능
print(lr.score(test_poly , test_target)) #-74 사용불가
print(lr.score(train_poly , train_target))

# 규제 하기 위한 전처리(스케일링)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scale = ss.transform(train_poly)
test_scale = ss.transform(test_poly)

# 릿지/라쏘 회귀들은 과적합된 자료들을 자동으로 특성을 제거해준다
# 릿지 회귀 : 가중치 줄여가면서
# 알파(alpha = 규제단위) , alpha단위가 크면 클수록 가중치(기울기)를 0으로 가깝게 만든다 
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scale , train_target)

alpha_list = [0.001 , 0.01, 0.1 , 1 , 10 , 100 ]
for alpth in alpha_list :
    ridge = Ridge(alpha=alpth)
    ridge.fit(train_scale , train_target)
    print("-----------------------------------" , alpth)
    print(ridge.score(train_scale , train_target))
    print(ridge.score(test_scale , test_target))



# 라쏘 회귀 : 자료 특성 간의 관계없는 특성들을 제거 하는 목적
# 특정한 특성의 값을 변경했을때 결정계수의 오차가 거의 없으면 관계없다고 판단
# 관계 없는 특성은 0으로 제거한다
# 예) 길이가 50->30으로 줄였을 때 성능/오차 없다 = 필요없는 특성 0으로 변경
# 예) 넓이 50 -> 30으로 줄었을 때 성능/오차 있다 = 필요한 특성 그대로 유지

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=10)
lasso.fit(train_scale , train_target) # 라쏘 모델 학습
print(lasso.score(train_scale , train_target))
print(lasso.score(test_scale , test_target))