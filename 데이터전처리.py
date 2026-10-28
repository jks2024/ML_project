import numpy as np

# 길이와 무게 데이터
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7,
               31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5,
               34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0,
               38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 10.5, 10.6, 11.0, 11.2,
               11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]

fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0,
               500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0,
               610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0,
               714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
               7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2,
               19.7, 19.9]

# 길이와 무게를 하나의 2차원 배열로 결합
fish_data = np.column_stack((fish_length, fish_weight))

# 타겟 데이터: 도미(1), 빙어(0)
fish_target = np.concatenate((np.ones(35), np.zeros(14)))  # 35개의 1 + 14개의 0 결합 함

# 훈련 세트와 테스트 세트로 데이터 나누기
from sklearn.model_selection import train_test_split

# stratify를 사용해 도미/빙어 비율 유지하면서 데이터 나누기
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42
)

print("훈련 데이터 크기:", train_input.shape)  # (36, 2)
print("테스트 데이터 크기:", test_input.shape)  # (13, 2)

# K-최근접 이웃 모델 만들기
from sklearn.neighbors import KNeighborsClassifier

# k-NN 분류기 객체 생성 (k=5 기본값)
kn = KNeighborsClassifier()

# 훈련 데이터로 모델 학습
kn.fit(train_input, train_target)

# 테스트 세트 정확도 출력
print("테스트 세트 정확도:", kn.score(test_input, test_target))  # 1.0

# 새로운 샘플 예측
# 새 생선 데이터 (25cm, 150g)
new_sample = [25, 150]

# 예측 결과 출력
print("새 샘플 예측:", kn.predict([new_sample]))  # [0.]

import matplotlib.pyplot as plt

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^', c='red', label='New Sample')
plt.xlabel('length')
plt.ylabel('weight')
plt.legend()
plt.show()

# 해결 방법: 표준화(Standardization)
# z = (x - 평균) / 표준편차
# - mean(평균): 데이터의 중심
# - std(표준편차): 각 데이터가 평균에서 얼마나 떨어져 있는지를 제곱 → 평균 → 다시 제곱근 한 값

# 표준화 적용
# 평균과 표준 편차 계산 (열 방향)
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

print("평균:", mean)
print("표준편차:", std)

# 훈련 데이터 표준화
train_scaled = (train_input - mean) / std

# 테스트 데이터도 동일 기준으로 변환
test_scaled = (test_input - mean) / std

# 새로운 샘플에 대해서도 변환
new_scaled = (np.array([25, 150]) - mean) / std

# 모델의 재학습 및 예측
kn.fit(train_scaled, train_target)

# 테스트 세트 정확도 확인
print("표준화 후 정확도:", kn.score(test_scaled, test_target))  # 1.0

# 새 샘플 예측
print("새 샘플 예측:", kn.predict([new_scaled]))  # [1.], 길이와 무게 특성을 동일한 비율로 맞춰서 공평한 계산이 되도록 함

# 새 샘플의 이웃 확인
distances, indexes = kn.kneighbors([new_scaled])

# 이웃 샘플 시각화
plt.scatter(train_scaled[:,0], train_scaled[:,1], label='Train Data')
plt.scatter(new_scaled[0], new_scaled[1], marker='^', c='red', label='New Sample')
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D', c='green', label='Nearest Neighbors')
plt.xlabel('length (standardized)')
plt.ylabel('weight (standardized)')
plt.legend()
plt.show()




