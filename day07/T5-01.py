# ==========================================
# 데이터 준비 (3차원 특성: 무게, 당도, 단단함)
# ==========================================
import pandas as pd

data = {  
    'weight': [110, 160, 130, 320, 370, 300, 55, 65, 60, 210, 220, 200, 90, 80, 100, 190, 180, 170, 100, 90,
               140, 280, 320, 130, 200, 140, 250, 150, 70, 80, 200, 300, 220, 140, 180, 230, 220, 250],
    'sweetness': [6.2, 7.2, 6.8, 8.1, 8.6, 8.1, 5.2, 5.7, 6.1, 7.2, 7.6, 6.7, 7.3, 6.9, 7.3, 7.5, 7.4, 7.3, 7.0, 6.8,
                  6.9, 8.0, 8.1, 6.7, 7.0, 6.6, 7.8, 7.1, 6.7, 6.5, 7.0, 7.6, 7.3, 7.0, 7.2, 7.5, 7.4, 7.7],
    'hardness': [7.8, 6.5, 7.1, 4.2, 3.5, 3.9, 8.9, 8.4, 8.1, 5.8, 5.2, 6.1, 7.3, 7.5, 7.0, 5.9, 6.2, 6.4, 7.2, 7.6,
                 6.8, 4.5, 4.1, 7.0, 5.7, 6.9, 4.9, 6.6, 8.2, 8.5, 5.8, 4.0, 5.3, 6.7, 6.1, 5.0, 5.2, 4.7]
}
df = pd.DataFrame(data)

# 테스트용
newDf = pd.DataFrame({'weight': [110], 'sweetness': [7.0], 'hardness': [7.5]})
features = ['weight', 'sweetness', 'hardness']

# 
from sklearn.cluster import KMeans
# n_clusters = k , 그룹 수 설정 , 2이면 2가지의 그룹으로 군집화 한다
# 
km = KMeans(n_clusters=2 , random_state=42) # 모델(비지도)학습 # target(정답/레이블)이 없다
km.fit(df[features])
print(km.predict(newDf[features]))# 모델 예측(클러스터/군집화)
print(km.labels_) # 행 마다의 군집 번호 , 0: 그룹A  , 1: 그룹B

# 시각화
import matplotlib.pyplot as plt
plt.scatter(df['weight'] , df['sweetness'] , c=km.labels_)
plt.scatter(newDf['weight'] , newDf['sweetness'],marker='*')
plt.show()

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
scaleDf = ss.fit_transform(df[features])
scaleNerDf = ss.transform(newDf[features])

# 스케일링 이후 시각화
plt.scatter(scaleDf[: , 0] , scaleDf[: , 1] , c=km.labels_)
plt.scatter(scaleNerDf[: , 0] , scaleNerDf[: ,1] , marker='*')
plt.show()

# 최적의 K(그룹 수) 찾기 , 엘보우 방법
sse = []
for k in range(1,11):
    km = KMeans(n_clusters=k , random_state=42) 
    km.fit(scaleDf)
    sse.append(km.inertia_) # 군집/그룹 내 자료들 간 오차의 제곱합 측정
print(sse) # 클러스터가 많아 지면 오차의 제곱합이 줄어든다

# 오차 시각화
plt.plot(range(1 ,11), sse , marker = 'o')
plt.show() 
# 엘보우 포인트 : SSE(오차의 제곱합) 급격하게 줄어든 포인트 -> 최적의 K

km = KMeans(n_clusters=3 , random_state=42)
km.fit(scaleDf)
df['cluster'] = km.labels_ # 클러스터 결과물
print(df)

# 거리 예측/계산 (추론 계산식) , 유클리드 거리
# 클러스터 들의 중심점
center = km.cluster_centers_
print(center)

# 중심점에서 새로운 자료의 오차 계산
import numpy as np
result = np.sqrt(np.sum((center - scaleNerDf)**2 , axis=1))
print(result) # [1.26432524 3.25608991 0.97575994] # 클러스터 중심점에서 새로운 자료의 거리

#
print(km.predict(scaleNerDf))


# 가우시안 모델 , 군집확률
from sklearn.mixture import GaussianMixture
# Kmaans 유사하게 정규분포의(군집)의 수
gm = GaussianMixture(n_components=3 , random_state=42)
gm.fit(scaleDf)
print(gm.predict(scaleNerDf))
print(gm.predict_proba(scaleNerDf)*100)

# 시각화
plt.scatter(scaleDf[: , 0] , scaleDf[: ,1] , c=df['cluster'])
plt.scatter(scaleNerDf[: , 0] , scaleNerDf[: ,1] , marker='*')
plt.show()

# PCA : 주성분 분석 , 차원 축소  , 차원이 크면 시각화 불가능하다 , 주초 2차원/3차원 압축한다
from sklearn.decomposition import PCA
# 여러개 특성/성분을 가진 모델들을 2차원/3차원으로 변경 , 무게/당도/단단함 -> 2차원으로 변경
pca = PCA(n_components=2) 
# 주성분 만들기 ; 각 특성/성분 마다의 가중치 더해서 데이터 변동선 개선
# 예 pca = 무게*가중치 + 당도*가중치 + 단단함*가중치
pcaDf = pca.fit_transform(scaleDf)
print(pcaDf) # 행 = 데이터 수 , 열 = 성분 수

df['pca_x'] = pcaDf[:,0] # 첫번째 열을 제1주성분 # 데이터의 변동성을 가장 크게 설명하는 주성분
df['pca_y'] = pcaDf[: , 1] # 두번째 열을 제2주성분 # 두번째로 변동성이 가장 크게 설명하는 주성분

# 주성분의 가중치 확인
components = pca.components_
print(components)

# 예측할 값을 주성분 변셩
pcaNewDf = pca.transform(scaleNerDf)

plt.scatter(df['pca_x'] , df['pca_y'] , df['cluster'] , marker='*')
plt.scatter(pcaNewDf[:, 0], pcaNewDf[:, 1], marker='*', s=200)
plt.ylabel('pca_2')
plt.show()

# 분석모델 스코어 , 실루엣 스코어(분리/응집)
from sklearn.metrics import silhouette_score
# silhouette_score(자료 , 모델군집)
sc = silhouette_score(scaleDf , km.labels_)
print(sc)

# gm 평가 모델dptjsms  K-mean처럼 labels_ 속성이 없다 , 그래서 예측을 통한 군집도 구한다
sc = silhouette_score(scaleDf , gm.predict(scaleDf))
print(sc)

# 스코어 개선 : 1. 최적의 k , 2 PCA 주성분(가중치) , 3 이상치 제거 

# HDBSCAN 자동으로 최적의 K와 이상치 제거 모델
import hdbscan
# min_cluter_size= 최소 클러스터 개수
# min_samples= 클러스터들의 중심점이 되기 위한 최소 자료(샘플) 수
# prediction_data = Treu data예측 가능
hdb = hdbscan.HDBSCAN(min_cluster_size=2 , min_samples=2 , prediction_data=True)
hdb.fit(scaleDf)
print(hdb.labels_) # -1(이상치 샘플) 어디에서 속하지 않은 샘플
print(hdbscan.approximate_predict(hdb , scaleDf))
print(hdbscan.approximate_predict(hdb , scaleNerDf)) # 1 그룹 예측