# Numpy = Numerical Python. 수학적인 계산에 있어 빠르고 효율적으로 사용 가능.
# Axes = Dimensions(차원).
# ex) [[1,0,0], [0,1,2]] ==> an array with 2 axes(2개 행과 3개 열을 가진 2차원 배열).
# ex2) 3차원 공간에서의 점 (1,2,1) ==> 길이가 3인 axis
# axis = 단수, axes = 복수 표현

# NumPy에서의 Array 클래스 표현 = ndarray
# ndarray의 표현 종류)
# - ndarray.ndim: axes의 숫자. 몇 차원인지
# - ndarray.shape: 각 axes마다 몇 개의 원소를 담을 수 있는지
# - ndarray.size:  원소에 포함된 전체 숫자
# - ndarray.dtype: 원소의 데이터 타입
# - ndarray.itemsize: 하나의 원소가 차지하는 byte 크기
# - ndarray.data: 각 실제 원소가 가진 버퍼량
# 
# ex) ndarray 표현 써보기
import numpy as np
from numpy import pi
# a = np.array([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]])
# a = np.arange(15).reshape(3, 5)
# print(a)
# print(a.shape)
# print(a.ndim)
# print(a.dtype.name)
# print(a.itemsize)
# print(a.size)
# print(type(a))

# ex2)
# import numpy as np
# b = np.array([6, 7, 8])
# print(b)
# print(type(b))

# 1. numpy에서 array 생성하는 법: array 함수 사용하기. np.array([2,3,4])
# - 만약 np.array로 만든 배열 내부에 float형태 등의 element가 있는 경우, numpy가 알아서 인식하고
# 데이터타입을 변환해준다.
# ex)
# b = np.array([1.2, 3.4, 5.6])
# print(b.dtype)

# 2. np.array 내부에 대괄호로 표현을 하지 않으면 오류 발생!
# - 만약 np.array([(a, b, c), (e, f ,g)])와 같이 소괄호로 내부가 표현되어 있
# 더라도, 출력하게 되면 자동으로 대괄호로 바뀜.
# ex) 배열 괄호 자동 변경하기
# b = np.array([(1.2, 2.3, 4.4), (4,5,6)])
# print(b)
# 3. 배열에 특정한 데이터 타입으로 표현하고 싶으면, 배열 뒤에 dtype = ~~를 표시.
# ex2) array dtype 특정하기
# c = np.array([[1,2],[3,4]], dtype = complex)
# print(c)

# 4. 만약 Element의 값은 모르지만, 그들의 Size는 아는 경우:
#     - np.zeros(()) ==> 소괄호 안의 값의 배열로 0을 생성해 준다.
#     - np.ones(()) ==> 소괄호 안의 값의 배열로 1을 생성해 준다.
#         - array 뒤에 dtype을 작성해서 특정할 수 있다.
#     - np.empty(()) ==> 현재의 메모리값에 따라 내부 element가 달라진다.
# ex) np.zeros/ones/empty 써보기
# print(np.zeros((3,4)))
# print(np.ones((2,3,4),dtype=np.int16))
# print(np.empty((2,3)))

# 5. np.arange: 시작점과 끝나는 점, 그리고 사이의 얼마만큼의 step을 갈 지 결정
#     - np.arange(10, 30, 5)의 경우, 시작점 = 10, 끝점 = 30, 각 스텝 = 5.
#     - array([10, 15 ,20, 25])로 표현이 됨.
#     - 마지막 30이 출력되지 않는 이유: 파이썬에서는 마지막 끝값은 출력하지 않음.
#     - 만약 step이 진행되는 도중 끝점을 넘게 되면, 넘기 직전의 값까지만 출력.
#     - 만약 np.arange() 내부에 실수형태가 있는 경우, array는 모두 실수형태로 출력됨.
# ex) np.arange 써보기
# print(np.arange(1, 10, 3))
# print(np.arange(1, 8, 3))
# print(np.arange(1, 10.1, 3))

# 6. linspace(linear space): arange와 비슷하지만 다름
#     - np.linspace(a, b, c) ==> a와 b 사이에 c개의 숫자를 생성할 것.
#     - linspace는 arange와 달리, 끝점을 포함하여 출력하게 된다.
#     - linspace는 step 범위를 생성할 숫자에 맞춰 계산하여 나눈다.
#     - *원하는 수의 element를 만들 때 유용하며, 
#     - **많은 수의 점을 계산할 때도 유용하다. ex) xy 그래프 만들기
# ex) np.linspace 써보기
# from numpy import pi
# print(np.linspace(0,2,9))
# x = np.linspace(0, 2*pi, 100)
# f = np.sin(x)
# print(f)

# 7. 큰 크기의 array를 출력할 때:
# - numpy는 자동으로 가운데 부분을 출력 스킵하고(, ... 이런식), 앞뒤만 출력함.
# - **만약 전체 array가 보고 싶은 경우
# - np.set_printoptions(threshold=np.nan) 이렇게 출력하면 된다.
# ex) 큰 크기의 array 출력해 보기
# print(np.arange(10000).reshape(100,100))

# 8. Arithmetic Operations
# - elemental-wise 방식이 적용된다.
# 사칙연산 뿐만 아니라 제곱, Boolean 등도 적용된다.
# ex) array에 수학연산 해보기
# a = np.array([20,30,40,50])
# b = np.arange(4)
# c = a - b
# print(b)
# print(c)
# print(b**2)
# print(10*np.sin(a))
# print(a<35)

# 8-1. 행렬곱
# - *의 경우 각 array의 같은 위치의 element끼리 계산하는 element-wise가 적용,
# - 행렬곱에서는 @나 .dot()가 사용된다.
# ex) 행렬곱 @, .dot() 써보기
# A = np.array([[1,1],[0,1]])
# B = np.array([[2,0],[3,4]])
# print(A*B)
# print(A@B)
# print(A.dot(B))

# 8-2. 두 데이터타입이 다른 array의 연산
# - 정수 + 실수 = 실수 타입으로 출력
# - 
# ex) 다른 데이터타입끼리 연산하기
# from numpy import pi
# a = np.ones(3, dtype = np.int32)
# b = np.linspace(0, pi, 3)
# print(b.dtype)
# c = a + b
# print(c)
# print(c.dtype)
# d = np.exp(c*1j)
# # exp(x) = 지수 e의 x승, j = 복소수의 허수 부분
# print(d)
# print(d.dtype.name)

# 9. Unary Operation: 하나의 ndarray의 method를 연산하기
# - sum(), min(), max() 등이 있다
# ex) unary operation 써보기
# a = np.random.random((2, 3))
# print(a.sum())
# print(a.min())
# print(a.max())

# 9-1. 특정한 axis에 대해서만 Unary Operation이 하고 싶은 경우:
# - a.sum(axis=0)과 같이, axis를 행렬에 따라 각각 0, 1로 정해주고 표현 가능.
# ex) 특정 axis만 써보기
# b = np.arange(12).reshape(3,4)
# print(b)
# # axis = 0 --> 각 column
# # axis = 1 --> 각 row
# print(b.sum(axis=0))
# print(b.sum(axis=1))
# print(b.min(axis=0))
# print(b.min(axis=1))
# # cumsum: 각 행/열의 중첩된 합을 구하기
# print(b.cumsum(axis=0))
# print(b.cumsum(axis=1))

# 10. Universal Functions(ufunc)
# - 삼각함수, 지수함수, 루트 등
# ex) Universal Function 써보기
# B = np.arange(3)
# C = np.array([2., -1., 4.])
# print(np.exp(B))
# print(np.sqrt(B))
# print(np.add(B,C))

# 11. Indexing, Slicing and Iterating on Arrays
# - 특정한 부분의 array를 보거나 변경할 때 사용
# - 1차원 array는 list처럼 indexing, slicing 그리고 iterating될 수 있음
# ex) indexing/slicing/iterating 해보기
# a = np.arange(10)**3
# print(a)
# # indexing: 해당 순서의 배열값 확인하기
# print(a[2])
# # slicing: 해당 slicing한 부분의 배열값 확인하기
# print(a[2:5])
# # iterating: 해당 부분에 다른 값 대입하기
# a[:6:2] = -1000 #의미: a배열의 0번째부터 6번째 배열값까지, index 2칸마다 해당 값을 -1000으로 바꾼다.
# print(a)
# print(a[::-1]) #의미: a배열의 앞뒤 순서를 뒤바꾼다.

# 11-1. 다차원 배열에서의 indexing/slicing/iterating
# - 규칙: 하나의 axis당 하나의 index를 사용
# ex) 다차원 배열에서 indexing/slicing/iterating 해보기
# def f(x,y):
#     return 10 * x + y
# b = np.fromfunction(f, (5, 4), dtype = int)
# print(b)
# print(b[2,3])
# print(b[0:5, 1])    #의미: b배열의 0~5행에서 1열의 값들만 출력하기
# print(b[:, 1])  #의미: 1열에 해당하는 모든 행의 값들 출력하기
# print(b[1:3, :])    #의미: 1~3행까지의 모든 열의 값들 출력하기
# print(b[-1])    #의미: -1 = b배열의 가장 마지막 열 == b[-1, :]
# 특징)
# - b[i]에서 i는 axes가 있는 만큼 :을 사용한다 ex) x[1] = x[1,:,:,:,:]과 equivalent하다.
# - '...'를 사용할 수 있다. 즉, 도트 3개를 써서 축약할 수 있다.
# ex) x[1,2,...] = x[1,2,:,:,:]이고, x[...,3] = x[:,:,:,:,3], x[4,:,:,5,:] = x[4,...,5,:]

# 다차원 array에서 iterating하기: 
# - 다차원에 있는 모든 원소를 접근하려 하는 경우: flat을 사용하여 해당 배열을 일렬로 길게 늘린다.
# ex) for element in b.flat:
#         print(element) 


# 12. Changing Array Shape: 각 Axes마다 얼마나 많은 index가 들어가 있는가
# ***배열 자체가 바뀌는 게 아니라, 배열의 리턴값이 바뀌는 것!!!!!
# 12-1. Ravel: 2차원의 배열을 1차원으로 펴줌.
# 12-2. Reshape: 배열의 차원을 원하는 모양으로 바꿔줌.
# 12-3. Transpose: 행과 열을 서로 바꿔줌.
# ex) 배열 모양 바꿔보기
# a = np.floor(10*np.random.random((3, 4)))   #실수 형태가 기본적으로 출력됨.
# print(a)
# print(a.shape)
# print(a.ravel())
# print(a.reshape(6,2))
# print(a.T)
# print(a.T.shape)

# 12-1-1. Ravel vs Flatten
# - 공통점: 다차원 배열을 1차원 배열로 반환해줌.
# - 차이점: 
#     - 1) flatten은 copy를 리턴함
#     - 2) ravel은 가능하면 기존 배열의 view를 리턴하고, 출력물에서는 보이지 않는다.
#     - 3) 만약 ravel을 통해 리턴한 배열을 바꾼다면, 기존 배열도 같이 수정된다
#     - 4) Flatten은 기존 배열이 같이 수정되지 않는다
#     - 5) Ravel이 더 빠르고 메모리 사용도 적다
# ex) Ravel vs Flatten
# a = np.floor(10*np.random.random((3, 4)))   #실수 형태가 기본적으로 출력됨.
# print(a)
# b = a.ravel()
# print(b)
# b[0] = 20.0 #b의 값을 바꾸면, a의 배열값도 같이 바뀐다.
# print(b)
# print(a)
# c = a.flatten()
# print(c)
# c[0] = 2.0  #c의 값을 바꿔도, 기존 a의 값은 바뀌지 않는다.
# print(c)
# print(a)

# 13. 배열 Size 바꾸기: Resize
# - ndarray.resize ==> 원본 배열을 직접적으로 바꿔준다.
# - 만약 axes 자리에 -1이 있는 경우: 다른 차원들은 자동으로 계산된다.
# - 예시) 만약 12개짜리 배열값을 가진 배열이 존재하고, reshape(a, -1)을 작성한다면,
# - numpy가 자동으로 12/a 만큼의 열을 계산해서 reshape해준다.
# - axes가 많거나 계산이 복잡한 경우에 유용하다.
# ex) resize 해보기
# a = np.floor(10*np.random.random((3, 4)))   #실수 형태가 기본적으로 출력됨.
# print(a)
# a.resize((2,6))
# print(a)
# print(a.reshape(3, -1))

# 14. 배열 Stacking하기: vstack & hstack
# - vstack = 수직으로 쌓기, hstack = 수평으로 쌓기
# ex) stacking 해보기
# a = np.floor(10*np.random.random((2,2)))
# print(a)
# b = np.floor(10*np.random.random((2,2)))
# print(b)
# print(np.vstack((a,b))) # a가 위, b가 아래 쌓임
# print(np.hstack((a,b))) # a가 왼쪽, b가 오른쪽에 쌓임

# 14-1. 1차원 배열 stacking: column_stack & row_stack
# - column_stack: 각 배열의 배열값들을 column처럼 취급하여 각 배열을 세로배열한 뒤 이어붙임.
#                 ==> 1차원 배열을 2차원 배열로 출력함!
#                 - hstack의 경우 1차원 배열이 1차원 배열로 출력됨.
#                 - 기존 배열이 '2차원'인 경우에만 동일한 결과가 출력됨.
# - row_stack: 배열값을 row취급하여 각 배열을 가로배열한 뒤 이어붙임.
#                 - vstack과 동일함.
# ex) column_stack vs hstack
# from numpy import newaxis   # newaxis = 새로운 axis를 '하나 더' 만들 수 있게 함.
# a = np.array([4., 2.])
# b = np.array([3., 8.])
# print(np.column_stack((a,b)))
# print(np.hstack((a,b))) # column_stack과 hstack의 결과값이 다름
# print(a[:, newaxis])   # 기존 a[:]의 경우 공간이 차있지만, newaxis를 입력함으로써 2차원이 되게 함.
# print(np.column_stack((a[:,newaxis],b[:,newaxis])))
# print(np.hstack((a[:,newaxis],b[:,newaxis])))   # 2차원의 경우 column_stack과 hstack 결과가 같음.

# 15. Newaxis: 새로운 차원 하나 더 만들기
# ex) 1차원 배열에서 새로운 차원 하나 더 만들기
# arr = np.arange(4)
# print(arr)
# print(arr.shape)
# # - 만약 새롭게 2차원 배열의 row를 만들고 싶은 경우:
# row_vec = arr[np.newaxis,:]
# print(row_vec)
# print(row_vec.shape)
# # - 만약 새롭게 2차원 배열의 column을 만들고 싶은 경우:
# col_vec = arr[:, np.newaxis]
# print(col_vec)
# print(col_vec.shape)

# 16. 배열 나누기: hsplit, vsplit
# ex) 배열을 수평(horizontal)/수직(vertical)으로 나눠보기
# a = np.floor(10*np.random.random((2,12)))
# print(a)
# # - hsplit으로 나누기
# print(np.hsplit(a, 3))  # a를 일정하게 3등분하기
# print(np.hsplit(a,(3,4)))   # 사용자가 지정한 a의 3,4행을 기준으로 나누기
# - vsplit = 배열을 세로로 나누기
# - array_split: 사용자가 지정한 특정 축을 기준으로 나누기

# 17. 객체 참조
# - copy하지 않는 경우:
#     - 간단한 assignment의 경우 배열 객체를 copy하지 않음
#     - ex)
# a = np.arange(12)
# b = a   # 어떠한 새로운 객체도 생성되지 않음
# b is a  # a와 b는 같은 ndarray 객체를 참조하게 됨.
# b.shape = 3, 4  # b의 shape을 바꾸면, a도 같이 바뀌게 됨
# a.shape --> (3, 4) 가 됨.
#     - 파이썬은 mutable 객체를 reference로 전달하여 함수가 copy를 만들지 않는다.
#     - ex)
# def f(x):
#     print(id(x))    # f(x) = x의 주소(고유식별자)를 출력하는 함수
# id(a)   
# f(a)    # f(a)와 id(a)는 같은 주소를 참조하게 됨.

# 17-1. View(Shallow Copy): 같은 데이터를 공유하는 배열 객체.
# - 같은 데이터를 공유하지만, 새로운 배열 객체를 생성한다
# ex)
# c = a.view
# c is a  --> False   # c는 a의 view를 '가리키는' 것이므로 a와 동일하지는 않음
# c.base is a --> True   # 하지만 c의 base와 a는 같다.
# c.flags.owndata --> False   # c가 스스로의 저장소를 가지고 있지 않음
# c.shape = 2, 6  # c의 모양을 변화시켜도, a의 모양은 변하지 않는다.
# a.shape  --> (3, 4)
# c[0, 4] = 1234  # 하지만, 데이터는 변화시킬 수 있다.
# - 배열을 Slicing하는 경우, view는 자동으로 생성된다.
# ex)
# s = a[:, 1:3]   # a배열에서 모든 행의 1~3열을 slicing한 배열을 s라고 둠
# s[:] = 10   # s[:] = s에 대한 view를 생성. 1~3열의 값들을 모두 10으로 변경하게 됨.

# 17-2. Deep Copy: 실제 데이터 copy함. 배열과 데이터의 독립된 값들을 생성.
# ex)
# d = a.copy()    # d는 a와 별개의 복사본이 생성된 것.
# d is a  --> False   # d는 a와 동일하지 않음
# d.base is a --> False   # d의 뿌리도 a와 다름.
# d[0,0] = 9999   --> # a에 적용되지 않는다.

# 18. Linear Algebra(선형대수학) 함수 몇 가지
# - a.transpose(): 아까 .T와 같은 기능(2차원 이상의 차원에서)
# - np.linalg.inv(a): 기존 행렬의 inversion --> identity 행렬을 만들어줌.
# - np.eye(2): identity 행렬을 만듦.
# - i @ i: 행렬곱.


















