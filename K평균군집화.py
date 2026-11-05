import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 데이터 로딩
fruits = np.load('fruits_300.npy')
# 2차원 벡터로 변환
fruits_2d = fruits.reshape(-1, 100*100)  # 3차원 -> 2차원

# 클러스터 수를 바꿔가면서 inertia 측정
inertia = []

for k in range(2, 7):  # k = 2 ~ 6
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)

# inertia 시각화
plt.plot(range(2, 7), inertia, marker='o')
plt.xlabel('k (클러스터 수)')
plt.ylabel('inertia (클러스터 내 거리 제곱합)')
plt.title('Elbow Method to Find Optimal k')
plt.grid(True)
plt.show()

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
labels = km.labels_

def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n / 10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols,
                            figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            idx = i*10 + j
            if idx < n:
                axs[i, j].imshow(arr[idx], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()

# 7. 클러스터별 이미지 출력 (예: label 0번)
for i in range(3):
    print(f"클러스터 {i}번 이미지 예시:")
    draw_fruits(fruits[labels == i])