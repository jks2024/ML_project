import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class TitanicSurvivalPredictor:
    def __init__(self):
        self.model = None
        self.features = ['Pclass', 'Age', 'SibSp', 'Parch', 'FamilySize', 'Sex_male', 'Embarked_Q', 'Embarked_S']

    def train_model(self):
        # 데이터 로드 및 전처리 (교안 개요서의 데이터 처리 역량 반영)
        df = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv')
        # 전처리 기준은 훈련 행에서만 계산합니다 (뒤의 8:2 분리와 동일).
        from sklearn.model_selection import train_test_split
        _train_rows, _test_rows = train_test_split(
            df.index, test_size=0.2, random_state=42, stratify=df['Survived']
        )
        age_mean = df.loc[_train_rows, 'Age'].mean()
        embarked_mode = df.loc[_train_rows, 'Embarked'].mode().iloc[0]
        df['Age'] = df['Age'].fillna(age_mean)
        df['Embarked'] = df['Embarked'].fillna(embarked_mode)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

        # 범주형 변수 인코딩
        df['Sex'] = pd.Categorical(df['Sex'], categories=['female', 'male'])
        df['Embarked'] = pd.Categorical(df['Embarked'], categories=['C', 'Q', 'S'])
        df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

        X = df[self.features]
        y = df['Survived']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 로지스틱 회귀 모델 학습
        self.model = LogisticRegression(max_iter=500)
        self.model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        print(f"모델 정확도: {accuracy * 100:.2f}%")

    def predict(self, new_data):
        new_passenger = pd.DataFrame(new_data)
        survival_prob = self.model.predict_proba(new_passenger)
        return survival_prob[0][1], survival_prob[0][0]  # 생존 확률, 사망 확률 반환