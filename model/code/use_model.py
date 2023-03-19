import pickle
import re
from konlpy.tag import Okt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# 위의 코드는 저장된 Tokenizer 객체를 불러오는 부분이고, Tokenizer 객체를 훈련코드에서 저장해줘야 합니다.
# 훈련코드에서 거의 마지막 부분에 load_model('best_model.h5') 전후에 아래 코드를 넣어주세요.
# 그럼 Tokenizer 객체에 담긴 어휘 분석 정보가 현재 폴더의 tokenizer.pickle에 저장됩니다.
# 아래는 훈련코드에 추가할 내용
# import pickle
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle)


# 리뷰 예측해 보기
def sentiment_predict(s):
    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍',
                 '과', '도', '를', '을', '으로', '자', '에', '와', '한', '하다', '랑']

    okt = Okt()

    max_len = 30
    loaded_model = load_model('model/best_model.h5')

    with open('model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    s = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", s)
    s = re.sub('^ +', "", s)

    s = okt.morphs(s, stem=True)  # 토큰화
    s = [word for word in s if not word in stopwords]  # 불용어 제거

    encoded = tokenizer.texts_to_sequences([s])  # 정수 인코딩
    padded = pad_sequences(encoded, maxlen=max_len)  # 패딩
    score = float(loaded_model.predict(padded))
    print(score)
    return str(score)
    # if (score > 0.5):
    #     print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
    # else:
    #     print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))


# sentiment_predict('조금 재밌음')
# sentiment_predict("많이 재밌음")
# sentiment_predict("이딴게 영화냐 ㅉㅉ")
# sentiment_predict("감독 뭐하는 놈이냐?")
# sentiment_predict('와 개쩐다 정말 세계관 최강자들의 영화다')
