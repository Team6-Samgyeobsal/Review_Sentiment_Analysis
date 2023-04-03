
import numpy as np
import pandas as pd
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle


train_data = pd.read_csv(
    filepath_or_buffer="model/data/kakao_review_content2.csv", sep=",")
test_data = pd.read_csv(
    filepath_or_buffer="model/data/kakao_review_content.csv", sep=",")

print('데이터 불러오기')
print(train_data.shape)
print(test_data.shape)

# 중복 데이터 제거
train_data.drop_duplicates(subset=['document'], inplace=True)
test_data.drop_duplicates(subset=['document'], inplace=True)

print('중복 제거된 train data >> ', len(train_data))
print('중복 제거된 test data >> ', len(test_data))

# Null 값이 존재하는 행 제거
train_data = train_data.dropna(how='any')
test_data = test_data.dropna(how='any')
print('null 값 제거')
print(len(train_data))
print(len(test_data))

# 정규 표현식을 사용해 한글만 남기고 제거
train_data['document'] = train_data['document'].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
test_data['document'] = test_data['document'].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)

# white space data나 empty data를 Null으로 변경하고 제거
train_data['document'] = train_data['document'].str.replace(
    "^ +", "", regex=True)
train_data['document'].replace('', np.nan, inplace=True)
train_data = train_data.dropna(how='any')

test_data['document'] = test_data['document'].str.replace(
    "^ +", "", regex=True)
test_data['document'].replace('', np.nan, inplace=True)
test_data = test_data.dropna(how='any')

print('데이터 전처리 (한글만 남게끔)')
print(len(train_data))
print(len(test_data))

# 불용어
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘',
             '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

# KoNLPy에서 제공하는 형태소 분석기
# stem 옵션으로 정규화 수행
okt = Okt()

# 불용어 제거
X_train = []
for idx, sentence in enumerate(train_data['document']):
    # if (idx == 1000):  # 임시
    #     break
    temp_X = okt.morphs(sentence, stem=True)
    temp_X = [word for word in temp_X if not word in stopwords]
    X_train.append(temp_X)

X_test = []
for idx, sentence in enumerate(test_data['document']):
    # if (idx == 1000):  # 임시
    #     break
    temp_X = okt.morphs(sentence, stem=True)
    temp_X = [word for word in temp_X if not word in stopwords]
    X_test.append(temp_X)

print("데이터 정규화")


# 정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# 전체 훈련 데이터에서 등장 빈도가 높은 순서대로 부여됨

threshold = 3
total_cnt = len(tokenizer.word_index)  # 단어의 수
rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if (value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 개수 :', total_cnt)
print('%s번 이하로 등장하는 단어의 수: %s' % (threshold - 1, rare_cnt))
print("희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)


# 2번 이하로 등장하는 단어가 전체 단어의 55%나 차지하지만 등장 비율은 2%도 안됨 ==> 제거
vocab_size = total_cnt - rare_cnt + 1
print(vocab_size)

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

# 생성한 단어 사전 저장
with open('model/tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

# 단어 사전 딕셔너리 저장
with open('model/word_index.pickle', 'wb') as f:
    pickle.dump(tokenizer.word_index, f, protocol=pickle.HIGHEST_PROTOCOL)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

# 빈 샘플 제거
# drop_train = [index for index, sentence in enumerate(
#     X_train) if len(sentence) < 1]

# print("drop_train", drop_train)
# X_train = np.delete(X_train, drop_train, axis=0)
# y_train = np.delete(y_train, drop_train, axis=0)
# print(len(X_train))
# print(len(y_train))

# 패딩으로 샘플들 길이 동일하게
print('리뷰 최대 길이 >> ', max(len(l) for l in X_train))
print('리뷰 평균 길이 >> ', sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

max_len = 30
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)


# LSTM 으로 분류
model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('model/best_model.h5', monitor='val_acc',
                     mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[
                    es, mc], batch_size=60, validation_split=0.2)

loaded_model = load_model('model/best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
