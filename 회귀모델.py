# 분류(Classification): 데이터를 카테고리로 구분
# 회귀(Regression): 회귀 분석은 단순히 값을 예측하는 것을 넘어,
#                  입력 특성(독립 변수)과 타깃 값(종속 변수) 사이의 관계를 찾아내고 모델링하는 데 중점
import numpy as np  # 파이썬은 기본적으로 배열이 없어서 라이브러리 추가 필요

# 농어에 대한 리스트 데이터를 넘파이 배열로 변환
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

from sklearn.model_selection import train_test_split  # 훈련 세트와 테스트 세트를 나누는 함수

# 훈련 세트와 테스트 세트로 나눕니다.
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)

# 훈련 세트와 테스트 세트를 2차원 배열로 바꿉니다.
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# K-NN 최근접 이웃 회귀 모델 가져오기
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=3)  # 객체 생성, 이웃의 개수를 3개로 지정
knr.fit(train_input, train_target)

print(knr.predict([[50], [40], [30]]))

# 선형회귀: 예측하고자 하는 상황을 수치로 입력하면, 예측 결과를 수치로 출력하는 알고리즘
# 선형 회귀는 최소제곱법으로 잔차 제곱합을 최소화하는 기울기와 절편을 찾아, 산포된 데이터를 가장 잘 통과하는
# ‘최적의 직선(회귀선)’을 그려 주는 방법
from sklearn.linear_model import LinearRegression  # 선형회귀 모델 가져오기
lr = LinearRegression()

# 선형 회귀 모델을 훈련합니다.
lr.fit(train_input, train_target)

# 50cm 농어에 대해 예측합니다.
print(lr.predict([[50], [55], [60]]))

import matplotlib.pyplot as plt
# 훈련 세트의 산점도를 그립니다.
plt.scatter(train_input, train_target)
# 15에서 50까지 1차 방정식 그래프를 그립니다.
plt.plot([15, 60], [15*lr.coef_+lr.intercept_, 60*lr.coef_+lr.intercept_])

# 50cm 농어 데이터
plt.scatter(50, 1241.8, marker='^')
plt.scatter(55, 1436.9, marker='^')
plt.scatter(60, 1632.0, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 다항 회귀 (특성 추가)
# 무게 = a * 길이^2 + b * 길이 + c
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.predict([[50**2, 50], [55**2, 55], [60 ** 2, 60]]))

# 구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만듭니다.
point = np.arange(15, 60)

# 훈련 세트이 산점도를 그립니다.
plt.scatter(train_input, train_target)

# 15에서 49까지 2차 방정식 그래프를 그립니다.
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)

# 50cm 농어 데이터
plt.scatter(50, 1574, marker='^')
plt.scatter(55, 1999, marker='^')
plt.scatter(60, 2474, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()







