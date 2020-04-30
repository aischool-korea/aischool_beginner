name_to_age = {"Jenny": 20, "Ella":31}
name_to_age["John"] = 26
name_to_age["Tom"] = 41

print(name_to_age)
print(name_to_age["Ella"])
print(name_to_age.get("Ella"))

name_to_age = {}
name_to_age["Kangmin"] = 31
name_to_age["John"] = 26
name_lsit = ["kangmin", "John"]
print(name_to_age)
print(name_to_age["Kangmin"])

name_to_age["John"] = 26
name_to_age["Tom"] = 29
print(name_to_age)
print(name_to_age["Jenny"])
print(name_to_age["John"])
print(name_to_age["Tom"])

name_to_age["Jenny"] = 21
print(name_to_age["Jenny"])
print(name_to_age.get("Jenny"))

print(name_to_age.keys())

for name in name_to_age.keys():
    print(name, name_to_age[name])

for i, name in enumerate(name_to_age.keys()):
    print(i, name, name_to_age[name])

print("Andrew" in name_to_age)
print("Ella" in name_to_age)
