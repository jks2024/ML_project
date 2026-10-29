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


