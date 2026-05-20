import requests
import pandas as pd

# Spring Boot 전체 차량 데이터 조회 API
url = "http://localhost:8080/api/car"

response = requests.get(url)

# 응답 상태 확인
print(response.status_code)

# JSON 데이터 받기
data = response.json()

# pandas DataFrame으로 변환
df = pd.DataFrame(data)

print(df.head())
print(df.columns)



