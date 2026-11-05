# 결정트리: 데이터를 “예/아니오” 식으로 계속 나누며 학습하는 구조
# - 루트 노드부터 리프 노드까지 질문을 통해 결과에 도달하는 스무고개식 알고리즘
# - 대표적으로 설명 가능한 모델
from sklearn.datasets import load_wine    # 사이킷런 라이브러리에 포함된 데이터 셋을 가져 옴
from sklearn.model_selection import train_test_split

wine = load_wine()
# 훈련 세트와 테스트 세트 분리, 비율 8:2, 재현 가능하도록 시드값 고정
train_input, test_input, train_target, test_target = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# 결정 트리 모델 훈련
from sklearn.tree import DecisionTreeClassifier  # 결정 트리 분류 모델 가져 오기

dt = DecisionTreeClassifier(max_depth=3, random_state=42)  # 트리의 깊이를 3으로 제한
dt.fit(train_input, train_target)

# 트리 구조 출력
from sklearn.tree import export_text
print(export_text(dt, feature_names=wine.feature_names))

# 시각화
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
plot_tree(dt, filled=True, feature_names=wine.feature_names, class_names=wine.target_names)
plt.show()


# 특성 중요도 파악
for name, score in zip(wine.feature_names, dt.feature_importances_):
    print(f"{name:30}: {score:.3f}")



