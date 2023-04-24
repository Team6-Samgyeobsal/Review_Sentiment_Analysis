from flask import Flask, request
import sentimental_model.code.use_model as sentimental_model
import filtering_model.code.use_model as filtering_model

app = Flask(__name__)


@app.route('/api/review/sentimental')
def getScore():
    review = request.args.get('review')
    score = sentimental_model.sentiment_predict(review)
    possible = filtering_model.filtering_predict(review)
    response = possible
    if (response != -1):
        response = score
    print('===========================')
    print("review input : "+review)

    if (possible == 1):
        print("욕설이 아닙니다.")
        print('평점은 '+score+'점 입니다.')
    else:
        print("욕설 입니다.")
    print('===========================')

    return str(response)


# @app.route('/api/review/filtering')
# def getStore():
#     review = request.args.get('review')
#     return filtering_model.filtering_predict(review)


if __name__ == "__main__":
    app.run(debug=True, port=4000)
