# 로지스틱 회귀
# - 회귀 모델이지만 이진 분류에 주로 사용
# - 입력값을 0과 1 사이의 확률로 변환하는 시그모이드 함수(S자 형태의 곡선)를 사용
import pandas as pd  # 엑셀과 유사하고 주로 데이터 분석에서 자주 사용 됨
from sklearn.model_selection import train_test_split  # 훈련 세트와 테스트 세트 분리하는 함수
from sklearn.preprocessing import StandardScaler  # 전처리, 정규화(표준화), 평균과 표준 편차를 이용해 스케일을 맞춤
fish = pd.read_csv("./perch.csv")  # 판다스의 read_csv() 함수를 이용해 csv 파일을 읽음

# 물고기의 무게, 길이, 대각선 길이, 높이, 너비 정보를 선택
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()  # 타깃인 물고기의 종류

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42
)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)  # 훈련 데이터를 동일한 스케일로 만듬
test_scaled = ss.transform(test_input)    # 테스트 데이터를 동일한 스케일로 만듬

# k-NN 이웃 분류
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=3)  # 이웃 개수를 3으로 변경
kn.fit(train_scaled, train_target)

print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))
print(kn.predict(test_scaled[:5]))
print(kn.predict_proba(test_scaled[:5]))

# 로지스틱 회귀 (이진 분류)
from sklearn.linear_model import LogisticRegression

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))

# 로지스틱 회귀 다중 분류
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
print(lr.predict(test_scaled[:5]))
print(lr.predict_proba(test_scaled[:5]))

# [[  0.00%,   1.35%,  84.13%,   0.03%,  13.57%,   0.67%,   0.26%]   -> 3번째 클래스 확률이 가장 높음 (84.13%)
#  [  0.00%,   0.26%,   4.39%,   0.00%,   0.73%,  94.62%,   0.00%]   -> 6번째 클래스 확률이 가장 높음 (94.62%)
#  [  0.00%,   0.00%,   3.41%,  93.48%,   1.50%,   1.60%,   0.00%]   -> 4번째 클래스 확률이 가장 높음 (93.48%)
#  [  1.09%,   3.40%,  30.55%,   0.66%,  56.66%,   0.01%,   7.62%]   -> 5번째 클래스 확률이 가장 높음 (56.66%)
#  [  0.00%,   0.04%,  90.40%,   0.24%,   8.95%,   0.24%,   0.13%]]  -> 번째 클래스 확률이 가장 높음 (90.40%)

