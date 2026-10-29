# 다중 선형회귀: 농어(perch)의 길이, 높이, 두께 등 여러 특성(feature)을 이용하여 무게(weight)를 예측하는 회귀 모델
# - 2개 이상의 입력 특성으로 하나의 연속적인 값을 예측
import pandas as pd
df = pd.read_csv("./perch_csv_data.txt")  # 입력 특성 데이터: 길이, 높이, 두께
perch_full = df.to_numpy()
print(perch_full)

import numpy as np
# 무게에 대한 타깃 데이터
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 훈련 / 테스트 세트 분리
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state=42)

# 다항 특성 변환
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=5, include_bias=False) # 절편항 제거, 5차까지 특성 생성
train_poly = poly.fit_transform(train_input)
test_poly = poly.transform(test_input)

# 선형 회귀 모델 훈련
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

# 규제 적용(Regularization): 모델이 훈련 데이터에 지나치게 맞춰져 새로운 데이터에 대한 예측 성능이 떨어지는 과적합(Overfitting)을 방지하기 위한 핵심 기법
# 릿지 규제: 릿지 규제는 제곱합을 사용하여 큰 계수를 억제
# 라쏘 규제: 계수의 절댓값 합을 사용하며, 일부 계수를 0으로 만들어 특성 선택 기능도 제공

from sklearn.preprocessing import StandardScaler
ss = StandardScaler() # 데이터를 표준화하기 위해 사용
ss.fit(train_poly)
train_scaled = ss.transform(train_poly) # 훈련 데이어의 표준화된 결과가 저장
test_scaled = ss.transform(test_poly)   # 테스트 데이터의 표준화된 결과가 저장

# 릿지 회귀: 계수의 제곱합을 패널티로 적용, 모든 특정을 조금씩 줄이는 방식
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.3)  # 클수록 규제 강함(과소 적합 우려). 작을 수록 규제 약함(과대적합 우려), 1이 기본값
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

# 라쏘 회귀: 계수의 제곱합을 패널티로 적용, 일부 계수를 0으로 만들어 특성 선택 효과 발생, 불필요한 특성은 자동 제거 됨
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

# K-NN 인접 회귀로 결과 확인 해보기
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=5)
knr.fit(train_scaled, train_target)

print(f"훈련 세트의 점수: {knr.score(train_scaled, train_target):.4f}")
print(f"테스트 세트의 점수: {knr.score(test_scaled, test_target):.4f}")




