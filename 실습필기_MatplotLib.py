import numpy as np
from matplotlib import pyplot as plt
import math #pi를 정의하기 위해 필요함.

# matplotlib: numpy와 파이썬을 기반으로 한 여러 시각화 데이터 패키지.
# pyplot: matplotlib 모듈, 유용한 style functions의 집합

# 1. Simple Line plot: 평면상의 선 그리기
# ex) 각도에 따른 sine wave 그리기.
#     - 참고: 가로의 각의 크기는 radian으로, 2pi가 360도에 대응된다.
# x = np.arange(0, math.pi * 2, 0.05)
# y = np.sin(x)
# plt.plot(x,y)   #plot(x, y) == 가로 x, 세로 y의 형태의 표면을 생성한다.
# plt.xlabel("angle") #xlabel() = 가로축 설명
# plt.ylabel("sine")  #ylabel() = 세로축 설명
# plt.title("sine wave")  #title() = 그래프 설명
# plt.show()  #show() = 결과 출력

# 2. Bar Chart: 막대기 그래프
# - 확실히 구분되는 카테고리별로 양을 표현할 때 유용함.
# ex) 컴퓨터 언어별 수강 학생 수 출력하기.
# langs = ['C', 'C++', 'Java', 'Python', 'PHP']
# students = [23, 17, 35, 29, 12]
# plt.bar(langs, students)    #bar(a, b) = 카테고리 a에 대한 통계값 b를 출력
# plt.show()

# ex2) Barchart 심화: 여러 배열들의 순서별 크기 비교하기
# data = [[30, 25, 50, 20], [40, 23, 51, 17], [35, 22, 45, 19]]
# X = np.arange(4)
# plt.bar(X + 0.00, data[0], color = 'b', width = 0.25)    #width = bar의 너비, X + 0.00 = bar의 각 위치 조정용
# plt.bar(X + 0.25, data[1], color = 'g', width = 0.25)   #X + 0.25를 통해 컬러 b의 bar보다 0.25칸 옆에 그려지게 함.
# plt.bar(X + 0.50, data[2], color = 'r', width = 0.25)
# plt.show()

# 3. Histogram: 어떠한 데이터의 양적인 분포. 어떤 값에 얼만큼의 샘플이 존재하는가?
# ex) 학생 수
# a = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])
# plt.hist(a, bins = [0, 25, 50 ,75, 100])    #hist(대상 array, bins=[배열 내 데이터를 나누는 기준])
# plt.title("histogram of result")
# plt.xticks([0, 25, 50, 75, 100])    #xticks = x축의 기준점을 설정.
# plt.xlabel('marks')
# plt.ylabel('no. of students')
# plt.show()

# 4. Pie Chart: 원형의 퍼센티지 비교용.
# - 절대적인 양보다는, 비율을 비교할 때 유용함.(vs Bar chart)
# ex) 언어별 학생 비율 비교
# langs = ['C', 'C++', 'Java', 'Python', 'PHP']
# students = [23, 17, 35, 29, 12]
# plt.pie(students, labels = langs, autopct='%1.2f%%')
# #pie(비율이 될 변수, 변수에 해당하는 조건, 퍼센티지 표시 방법)
# plt.show()

# 5. Scatter Plot: 각 데이터 샘플을 x/y에 대응되도록 점으로 배치함.
# - 변수들의 경향을 파악하는 데 유용함.
# ex) 남녀 학생별 점수 분포도
# girls_grades=[89,90,70,89,100,80,90,100,80,34]
# boys_grades=[30,29,49,48,100,48,38,45,20,30]
# grades_range=[10,20,30,40,50,60,70,80,90,100]
# plt.scatter(grades_range, girls_grades, color = 'r')    #여학생별 점수분포
# plt.scatter(grades_range, boys_grades, color = 'b')     #남학생별 점수분포
# plt.xlabel('Grades Range')
# plt.ylabel('Grades Scored')
# plt.title('Scatter Plot')
# plt.show()

# 6. Box Plot: 데이터 값의 분포를 요약해서 표시할 때 유용.
# - 최소값, 1/4지점, 중간값, 3/4지점, 최대값 표시 가능
# ex) Random하게 추출한 값들을 표현해보기
np.random.seed(10)  #프로그램 실행 때마다 똑같은 양의 결과를 보기 위함.
dataSet1 = np.random.normal(100, 30, 200)   
#random.normal = 평균이 100이고, 표준편차가 30인 정규분포로부터 200개의 데이터 추출
dataSet2 = np.random.normal(80, 20, 200)
# 평균 80, 표준편차 20인 정규분포로부터 200개의 데이터 추출
plotData = [dataSet1, dataSet2] #List로 두 데이터를 묶음
plt.boxplot(plotData)   #boxplot = 데이터를 그려줌
plt.show()








