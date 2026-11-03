from flask import Flask, render_template, request
from titanic_predictor import TitanicSurvivalPredictor

app = Flask(__name__)

# 서버 실행 시 모델을 한 번만 학습시켜 유지합니다.
predictor = TitanicSurvivalPredictor()
predictor.train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 웹 폼에서 데이터 수집 (콘솔의 input() 역할 대체)
        try:
            pclass = int(request.form['pclass'])
            age = float(request.form['age'])
            sibsp = int(request.form['sibsp'])
            parch = int(request.form['parch'])
            sex_male = int(request.form['sex'])
            embarked_q = int(request.form.get('embarked_q', 0)) # 체크박스 미선택 시 0
            embarked_s = int(request.form.get('embarked_s', 0))

            # 클래스 양식에 맞춰 데이터 구성
            new_data = {
                'Pclass': [pclass], 'Age': [age], 'SibSp': [sibsp], 'Parch': [parch],
                'FamilySize': [sibsp + parch + 1], 'Sex_male': [sex_male],
                'Embarked_Q': [embarked_q], 'Embarked_S': [embarked_s]
            }

            # 기존 클래스의 예측 메서드 호출
            s_prob, d_prob = predictor.predict(new_data)

            return render_template('index.html', prob=round(s_prob*100, 2), death=round(d_prob*100, 2))
        except Exception as e:
            return f"입력 오류 발생: {e}"

if __name__ == '__main__':
    app.run()