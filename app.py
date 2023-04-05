from flask import Flask, request
import sentimental_model.code.use_model as sentimental_model
import filtering_model.code.use_model as filtering_model

app = Flask(__name__)

@app.route('/api/review/sentimental')
def getScore():
    review = request.args.get('review')
    return sentimental_model.sentiment_predict(review)

@app.route('/api/review/filtering')
def getStore():
    review = request.args.get('review')
    return sentimental_model.sentiment_predict(review)


if __name__ == "__main__":
    app.run(debug=True, port=4000)
