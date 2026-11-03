# 1.데이터 로드 -> 2. 결측 처리 -> 3. 파생변수 생성 -> 4. 인코딩 ->5. 훈련/테스트 세트 분리 -> 학습 -> 평가 -> 예측
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv')

print(df.info())      # 데이터 타입에 대한 정보
print(df.head())      # 상위 5명 정보
print(df.describe())  # 통계적인 정보

# 데이터 전처리




