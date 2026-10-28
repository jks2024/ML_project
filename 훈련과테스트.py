# 지도학습: 입력 데이터와 함께 '정답(타깃/레이블)'이 제공
# - 새로운 데이터가 들어왔을 때 정답을 맞히는 것.
# 비지도학습: 정답 없이 '입력 데이터'만 제공
# - 데이터 자체의 통계적 구조나 숨겨진 패턴을 찾아내는 것

# 훈련 세트와 테스트 세트가 필요한 이유
# - 모델을 평가할 때 가장 큰 실수는 학습에 사용한 데이터로 성능을 측정하는 것 (정답을 알고 시험을 치르는 것과 동일)
# - 훈련 세트 (Training Set): 모델을 가르치는 '연습서'
# - 테스트 세트 (Test Set): 모델의 진짜 실력을 검증하는 '실전 시험'
# - 훈련세트와 테스트 세트를 나눌 때 샘플링 편향이 발생 할 수 있음에 유의
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 데이터 변환 및 레이블 생성
fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1] * 35 + [0] * 14  # 도미는 1, 빙어는 0

# 리스트에 대한 인덱싱 및 슬라이싱
# 1. fish_data의 5번 인덱스 값 출력
print(fish_data[5])
# 2. fish_data의 처음 다섯개 출력
print(fish_data[:5])
# 3. fish_data의 마지막 5개 출력
print(fish_data[-5:])

# 샘플링 편향 문제 확인
# ❌ 단순히 앞 35개를 훈련, 나머지를 테스트로 분리
train_input  = fish_data[:20]    # 도미 데이터만 들어감!
train_target = fish_target[:20]
test_input   = fish_data[10:]    # 빙어 데이터만 들어감!
test_target  = fish_target[10:]

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
print(kn.score(test_input, test_target))

# 랜덤 셔플로 샘플링 편향 해결
import numpy as np

# 리스트 → 넘파이 배열 변환
input_arr  = np.array(fish_data)    # shape: (49, 2)
target_arr = np.array(fish_target)  # shape: (49,)

# 인덱스 배열 생성 후 무작위 셔플
np.random.seed(42)       # 재현 가능한 결과를 위한 시드 고정
index = np.arange(49)    # [0, 1, 2, ..., 48]
np.random.shuffle(index) # 인덱스를 무작위로 섞음

print(index)

train_input  = input_arr[index[:35]]   # 섞인 인덱스 중 앞 35개 → 훈련 입력
train_target = target_arr[index[:35]]  # 섞인 인덱스 중 앞 35개 → 훈련 타깃
test_input   = input_arr[index[35:]]   # 섞인 인덱스 중 뒤 14개 → 테스트 입력
test_target  = target_arr[index[35:]]  # 섞인 인덱스 중 뒤 14개 → 테스트 타깃

# 데이터가 섞여 있는 정도를 시각화로 확인
import matplotlib.pyplot as plt
plt.scatter(train_input[:, 0], train_input[:,1]) # 모든 행의 0번째 열, 모든 행의 1번째 열
plt.scatter(test_input[:,0], test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 모델 훈련과 평가
kn = KNeighborsClassifier()  # KNN 최근접 이웃 분류 객체 생성
kn.fit(train_input, train_target)  # 모델 학습

# 모델 평가 및 결과 확인
train_score = kn.score(train_input, train_target)  # 훈련 세트 정확도
test_score = kn.score(test_input, test_target)  # 테스트 세트 정확도
print("훈련 세트 정확도:", train_score)
print("테스트 세트 정확도:", test_score)

# 새로운 데이터 예측
print(kn.predict([[30, 600], [10, 50], [100, 250]]))


