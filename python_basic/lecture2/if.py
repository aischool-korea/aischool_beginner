man = False

if man:
    print("남자화장실로 가세요")
else:
    print("여자 화장실로 가세요")


minimum = 165
height = 163

if height <= minimum:
    print("탑승하실수 없습니다")
else:
    print("탑승하세요")

blood_type = "B"
emergency_patient = "A"

if blood_type == emergency_patient:
    print("수혈해 주세요")
else:
    print("수혈해 주실수 없습니다")

minimum = 165
maximum = 195
height = 174
if height < minimum or height > maximum:
    print("탑승하실수 없습니다")
else:
    print("탑승하세요")

blood_type1 = "B"
emergency_patient_type1 = "A"
blood_type2 = "RH+"
emergency_patient_type2 = "RH+"

if blood_type1 == emergency_patient_type1 and blood_type2 == emergency_patient_type2:
    print("수혈해 주세요")
else:
    print("수혈해 주실수 없습니다")
basic = 40
intermediate = 70
advanced = 100
score = -10
if score <= basic:
    print("초급반을 수강하세요")
elif score <= intermediate:
    print("중급반을 수강하세요")
elif score <= advanced:
    print("고급반을 수강하세요")
else:
    print("점수를 확인해주세요")
