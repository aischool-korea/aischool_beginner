count = 0
while count < 10:
    count += 1
    print(count)

prompt = """
1. Add
2. Del
3. Quit"""
number = 0
while number != 3:
    print(prompt)
    number = int(input("Enter number:"))

coffee = 3
while True:
    money = int(input("돈을 넣어 주세요: "))
    if money == 300:
        print("맛있게 드세요.")
        coffee = coffee -1
    elif money > 300:
        print("거스름돈은 %d원입니다." % (money -300))
        print("맛있게 드세요.")
        coffee = coffee -1
    else:
        print("%d 더 넣어주세요." % (300 - money))
    if coffee == 0:
        print("커피가 다 떨어졌습니다. 판매를 중지 합니다.")
        break


coffee = 3
while coffee > 0:
    print(f'남은 커피: {coffee}')
    money = int(input("돈을 넣어 주세요: "))
    if money < 300:
        continue
    coffee -= 1
    print("맛있게 드세요.")


