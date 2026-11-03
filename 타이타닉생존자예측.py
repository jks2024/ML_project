# 1.데이터 로드 -> 2. 결측 처리 -> 3. 파생변수 생성 -> 4. 인코딩 ->5. 훈련/테스트 세트 분리 -> 학습 -> 평가 -> 예측
import pandas as pd
df = pd.read_csv('./titanic.csv')

# print(df.info())      # 데이터 타입에 대한 정보
# print(df.head())      # 상위 5명 정보
# print(df.describe())  # 통계적인 정보

# 데이터 전처리
# 결측치 확인
print(df.isnull().sum())

# 전처리 기준은 훈련 행에서만 계산합니다 (뒤의 8:2 분리와 동일).
from sklearn.model_selection import train_test_split

train_rows, test_rows = train_test_split(
    df.index, test_size=0.2, random_state=42, stratify=df['Survived']
)

age_mean = df.loc[train_rows, 'Age'].mean()                    # 나이에 대한 평균값 구하기
embarked_mode = df.loc[train_rows, 'Embarked'].mode().iloc[0]  # 탑승항구
df['Age'] = df['Age'].fillna(age_mean)   # 나이에 대한 평균값으로 결측치 채우기
df['Embarked'] = df['Embarked'].fillna(embarked_mode)  # 탑승항구 결측치 채우기

# 결측치 재확인
print(df.isnull().sum())

# 파생 함수 생성, 새로운 열 추가: 가족수(FamilySize)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print(df.head())

# 데이터 분석
# 생존자 비율
survival_rate = df['Survived'].mean()
print(f"Overall Survival Rate: {survival_rate * 100:.2f}%")

# 성별에 따른 생존자 비율
survival_rate_by_gender = df.groupby('Sex')['Survived'].mean()
print(survival_rate_by_gender)

# 클래스에 따른 생존자 비율
survival_rate_by_class = df.groupby('Pclass')['Survived'].mean()
print(survival_rate_by_class)

# 데이터 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# 성별에 따른 생존자 비율 시각화
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.show()

# 클래스에 따른 생존자 비율 시각화
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Class')
plt.show()

# 나이에 따른 생존자 비율 시각화
plt.figure(figsize=(10, 6))
sns.histplot(df[df['Survived'] == 1]['Age'], bins=30, label='Survived', kde=False, color='green')
sns.histplot(df[df['Survived'] == 0]['Age'], bins=30, label='Not Survived', kde=False, color='red')
plt.legend()
plt.title('Survival Rate by Age')
plt.show()

# 로지스틱 회귀 모델링
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 필요한 열 선택 및 더미 변수화
df['Sex'] = pd.Categorical(df['Sex'], categories=['female', 'male'])
df['Embarked'] = pd.Categorical(df['Embarked'], categories=['C', 'Q', 'S'])  # 1, 0
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)  # 남성이면 1, 여성이면 1

# 특성과 타겟 변수 정의
X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

# 훈련 세트와 테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 로지스틱 회귀 모델 훈련
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 정확도와 분류 보고서 출력
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))



