# 1. Numpy: 다차원 array를 구현하고 처리할 때 유용.
# 2. SciPy: 선형대수학, 통계 등 사용에 적합, NumPy와 연계
# 3. Pandas: DB의 테이블 사용에 적합, Series/Data Frames와 유사함.
# 데이터 처리에 적합한 함수들이 존재함. 
# 4. SciKit-Learn: 머신러닝 알고리즘을 제공, 모델링 예측 등에 사용
# 5. matplotlib: 2D/3D 플로팅 라이브러리. 히스토그램, 바 차트 등에 사용
# 6. Seaborn: matplotlib에 기반, 세부적인 것보단 high level 인터페이스.

# ndarray.ndim: dimension의 개수
# ndarray.shape: 모양
# ndarray.size, dtype, itemsize, data 등

# Matplotlib: 그래프를 쉽게 그려주는 라이브러리.
# NumPy를 이용해 다양한 차원의 그래프화.
# pyplot: Matplotlib의 그래프 명령 모듈.
# pylab: matplotlib.pyplot과 Numpy를 동시에 import하기 위한 함수

#과제: Numpy를 이용해 학생 100명의 BMI를 계산하기.
# 무게kg, 키m단위로 계산, 사이즈 100 짜리 wt와 ht어레이를 만들고,
# 40.0~90.0 사이의 wt어레이, 140~200 사이의 ht어레이(cm)단위
# BMI 100개를 계산하고 bmi 어레이에 저장한 뒤, 해당 어레이를 출력.

#과제 2. MatplotLib을 이용해 Bar chart(각 BMI 100개를 4개의 Status로
# 나눠 각 Status에 몇 명이 있는지 표현), Pie Chart(한 Status가 몇%인지)
# Scatter Plot(어떻게 height, weight이 분포되어 있는지)

import numpy as np
np.set_printoptions(precision = 1, suppress = True)
info = []
print("Input student's info.(weight, height):")
while True:
    wt = input("wt(kg): ")
    if wt == 'quit':
        break
    ht = input("ht(cm): ")
    if wt == 'quit':
        break
    wt = float(wt)
    ht = float(ht) * 0.01
    info.append([wt, ht])
info = np.array(info)
wt = info[:, 0]
ht = info[:, 1]
bmi = (wt / (ht ** 2))
print("bmi: ", bmi)

# import matplotlib.pyplot as plt
# status = []
# for value in bmi:
#     if value < 18.5:
#         status.append('Underweight')
#     elif value < 25:
#         status.append('Healthy')
#     elif value < 30:
#         status.append('Overweight')
#     else:
#         status.append('Obese')
# count = {'Underweight': 0, 'Healthy': 0, 'Overweight': 0, 'Obese': 0}
# for i in status:
#     count[i] += 1
# plt.bar(list(count.keys()), list(count.values()))
# plt.show()

# import matplotlib.pyplot as plt
# status = []
# for value in bmi:
#     if value < 18.5:
#         status.append('Underweight')
#     elif value < 25:
#         status.append('Healthy')
#     elif value < 30:
#         status.append('Overweight')
#     else:
#         status.append('Obese')
# count = {'Underweight': 0, 'Healthy': 0, 'Overweight': 0, 'Obese': 0}
# for i in status:
#     count[i] += 1
# order = ['underweight', 'healthy', 'overweight', 'obese']
# plt.hist(status, bins = 4)
# plt.title('BMI Histogram')
# plt.ylabel('Number of Students')
# plt.xlabel('BMI Level')
# plt.xticks(range(len(order)), order)
# plt.show()

# import matplotlib.pyplot as plt
# status = []
# for value in bmi:
#     if value < 18.5:
#         status.append('Underweight')
#     elif value < 25:
#         status.append('Healthy')
#     elif value < 30:
#         status.append('Overweight')
#     else:
#         status.append('Obese')
# count = {'Underweight': 0, 'Healthy': 0, 'Overweight': 0, 'Obese': 0}
# for i in status:
#     count[i] += 1

# plt.pie(count.values(), labels=count.keys(), autopct='%1.1f%%')
# plt.title('BMI Status')
# plt.axis('equal')
# plt.show()


import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.scatter(ht, wt)
plt.xlim(1.4, 2.0)
plt.title('Scatter Plot')
plt.xlabel('Height(m)')
plt.ylabel('Weight(kg)')
plt.show()