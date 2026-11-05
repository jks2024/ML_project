# 비지도 학습: 입력만 존재하고 정답은 제공하지 않음
# - 숨은 그룹이나 패턴을 찾아 냄(군집화 / 이상값 감지)
import numpy as np
import matplotlib.pyplot as plt

fruits = np.load("./fruits_300.npy")
print(fruits.shape) # 300개의 샘플 수, 높이가 100, 너비 100
print(fruits[0, 50, :])

# 이미지 확인 및 시각화
plt.imshow(fruits[0], cmap="gray")
plt.show()
plt.imshow(fruits[0], cmap="gray_r")
plt.show()

fig, axs = plt.subplots(1, 2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()

# 이미지 펼치기: 계산을 위한 준비
apple = fruits[0:100].reshape(-1, 100*100)  # 0 ~ 99까지를 형변환
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
print(apple.shape)
print(apple.mean(axis=1))

# 샘플별 픽셀 평균값 히스토그램
plt.hist(apple.mean(axis=1), alpha=0.8, label='apple')
plt.hist(pineapple.mean(axis=1), alpha=0.8, label='pineapple')
plt.hist(banana.mean(axis=1), alpha=0.8, label='banana')
plt.legend()
plt.show()

# 평균 이미지 만들기
apple_mean = apple.mean(axis=0).reshape(100, 100)
pineapple_mean = pineapple.mean(axis=0).reshape(100, 100)
banana_mean = banana.mean(axis=0).reshape(100, 100)
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()

# 대표 이미지와 가장 유사한 사진 찾기
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1, 2))
print(abs_mean.shape)

apple_index = np.argsort(abs_mean)[:100]
apple_index = apple_index.reshape(10, 10)
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i, j]], cmap='gray_r')
        axs[i, j].axis('off')
plt.show()

