from Python_basic.airtravel import Flight
#생성한 클래스를 import
f1 = Flight('KE081', 0)
f2 = Flight('KE082', 0)

f1.add_passenger(2)
f2.add_passenger(3)

print(f1.nation)
print(f2.nation)

# f1 = Flight('KE082', 0) #클래스 객체 생성 및 변수에 할당
# f2 = Flight('KE081', 0)
# f1.add_passenger(2)
# f2.add_passenger(3)
# print(f1._passenger_num)
# print(f2._passenger_num)
# print(f1.nation)
# print(f1.nation)

# print(type(f))

# print(f.number())
# f.__number = 'KE081'
# print(f.number())