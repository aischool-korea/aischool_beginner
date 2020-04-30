marks = [90, 25, 67, 45, 80, 75, 82]

number = 0
for mark in marks:
    number = number + 1
    if mark >= 60:
        print("%d번 학생은 합격" % number)
    else:
        print("%d번 학생은 불합격" % number)

marks = [90, 25, 67, 45, 80]

number = 0
for mark in marks:
    number = number + 1
    if mark < 60:
        continue
    print("%d번 학생 축하합니다. 합격입니다. " % number)

for i in range(10):
    print(i)

sum = 0
for i in range(1,11): #[1,2,3,4,5,6,7,8,9,10]
    sum += i
print(sum)

for i in range(2,10):#[2,3,4,5,6,7,8,9]
    for j in range(1, 10): #[1,2,3,4,5,6,7,8,9]
        print(i*j, end=",")
    print('')

# numbers = [1, 2, 3, 4, 5]
# result = []
# for n in numbers:
#     if n % 2 == 1:
#         result.append(n*2)
# print(result)
#
# result = [n*2 for n in numbers if n % 2 == 1]
# print(result)