from flask import Flask, request
import model.code.use_model as model

app = Flask(__name__)


@app.route('/api/review/sentimental')
def getScore():
    review = request.args.get('review')
    return model.sentiment_predict(review)


if __name__ == "__main__":
    app.run(debug=True, port=4000)

# TODO : 영화 리뷰가 아닌 맛집 리뷰 모델을 만들어야 됨
