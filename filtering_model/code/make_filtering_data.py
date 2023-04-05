import pandas as pd

import csv
import numpy as np

data = []

# 텍스트 파일을 배열로 읽어들임
with open('filtering_model/data/dev.hate.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for d, l in reader:
        if l == 'none':
            label = 0
        else:
            label = 1
        data.append([d, label])

# 배열 출력


with open('filtering_model/data/test.txt', 'w', newline='') as f:
    writer = csv.writer(f)
    for d in data:
        writer.writerow(d)
