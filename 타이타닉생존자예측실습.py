import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class TitanicSurvivalPredictor:
    def __init__(self):
        self.model = None  # 모델을 저장할 변수
        self.features = ['Pclass', 'Age', 'SibSp', 'Parch', 'FamilySize', 'Sex_male', 'Embarked_Q', 'Embarked_S']  # Fare 제외

    def train_model(self):
        # 1. 데이터 불러오기
        df = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv')

        # 2. 결측치 처리
        # 'Age' 열의 결측값을 평균값으로 채우기
        # 전처리 기준은 훈련 행에서만 계산합니다 (뒤의 8:2 분리와 동일).
        from sklearn.model_selection import train_test_split
        _train_rows, _test_rows = train_test_split(
            df.index, test_size=0.2, random_state=42, stratify=df['Survived']
        )
        age_mean = df.loc[_train_rows, 'Age'].mean()
        embarked_mode = df.loc[_train_rows, 'Embarked'].mode().iloc[0]
        df['Age'] = df['Age'].fillna(age_mean)

        # 훈련 세트의 최빈값을 공통 적용
        df['Embarked'] = df['Embarked'].fillna(embarked_mode)

        # 3. 새로운 열 추가: 가족 수(FamilySize)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

        # 4. 더미 변수화 (범주형 변수 처리)
        df['Sex'] = pd.Categorical(df['Sex'], categories=['female', 'male'])
        df['Embarked'] = pd.Categorical(df['Embarked'], categories=['C', 'Q', 'S'])
        df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

        # 5. 특성 및 타겟 정의 (Fare 제외)
        X = df[self.features]
        y = df['Survived']

        # 6. 훈련 및 테스트 데이터 분리
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 7. 로지스틱 회귀 모델 훈련
        self.model = LogisticRegression(max_iter=500)
        self.model.fit(X_train, y_train)

        # 8. 모델 평가 (테스트 데이터로 정확도 확인)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"모델 정확도: {accuracy * 100:.2f}%")

    def predict(self, new_data):
        # 새로운 승객 정보를 받아서 예측
        new_passenger = pd.DataFrame(new_data)

        # 예측 결과 반환 (predict_proba: 생존할 확률과 사망할 확률 반환)
        survival_prob = self.model.predict_proba(new_passenger)
        return survival_prob[0][1], survival_prob[0][0]  # 생존 확률, 사망 확률 반환


# 메뉴 함수
def menu():
    predictor = TitanicSurvivalPredictor()

    while True:
        print("\n1. 모델 학습")
        print("2. 새로운 승객 데이터 입력")
        print("3. 종료")

        choice = input("메뉴를 선택하세요: ")

        if choice == "1":
            # 1. 모델 학습
            print("모델을 학습 중입니다...")
            predictor.train_model()
            print("모델 학습 완료!")

        elif choice == "2":
            if predictor.model is None:
                print("모델이 학습되지 않았습니다. 먼저 '모델 학습'을 선택하세요.")
            else:
                # 2. 새로운 승객 데이터 입력
                try:
                    pclass = int(input("Pclass (1등석: 1, 2등석: 2, 3등석: 3): "))
                    age = float(input("Age (나이): "))
                    sibsp = int(input("SibSp (형제/배우자 수): "))
                    parch = int(input("Parch (부모/자녀 수): "))
                    sex_male = int(input("Sex (남성: 1, 여성: 0): "))
                    embarked_q = int(input("Embarked_Q (Q 항구 탑승: 1, 탑승하지 않음: 0): "))
                    embarked_s = int(input("Embarked_S (S 항구 탑승: 1, 탑승하지 않음: 0): "))

                    # 가족 수 계산
                    family_size = sibsp + parch + 1

                    # 새로운 승객 데이터 구성
                    new_passenger_data = {
                        'Pclass': [pclass],
                        'Age': [age],
                        'SibSp': [sibsp],
                        'Parch': [parch],
                        'FamilySize': [family_size],
                        'Sex_male': [sex_male],
                        'Embarked_Q': [embarked_q],
                        'Embarked_S': [embarked_s],
                    }

                    # 예측
                    survival_prob, death_prob = predictor.predict(new_passenger_data)

                    # 예측 결과 출력
                    print(f"\n생존 확률: {survival_prob * 100:.2f}%")
                    print(f"사망 확률: {death_prob * 100:.2f}%")
                except ValueError:
                    print("잘못된 입력입니다. 다시 시도해주세요.")

        elif choice == "3":
            # 3. 종료
            print("프로그램을 종료합니다.")
            break

        else:
            print("잘못된 선택입니다. 다시 시도해주세요.")


# 메뉴 실행
if __name__ == "__main__":
    menu()
