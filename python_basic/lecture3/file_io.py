f = open("./write.txt", 'w', encoding='utf-8')
f.write("file file file")
f.close()

f = open("./write.txt", 'w', encoding='utf-8')
for i in range(1, 100):
    data = f'line {i}\n'
    f.write(data)
f.close()

with open("./write.txt", 'w', encoding='utf-8') as f:
    for i in range(1, 10):
        data = f'line {i}\n'
        f.write(data)

f = open("./write.txt", 'a', encoding='utf-8')
for i in range(10, 20):
    data = f'line {i}\n'
    f.write(data)
f.close()

f = open("./write.txt", 'r', encoding='utf-8')
line = f.readline()
print(line)
f.close()

f = open("./write.txt", 'r', encoding='utf-8')
line = f.readline()
while line: # True ("line1"), False (None)
    print(line)
    line = f.readline()
f.close()

f = open("./write.txt", 'r', encoding='utf-8')
lines = f.readlines()
print(lines)
for line in lines:
    print(line.strip())
f.close()

f = open("./write.txt", 'r', encoding='utf-8')
content = f.read()
print(content)
f.close()

f = open("./write.txt", 'r', encoding='utf-8')
content = f.read(6)
print(content)
content = f.read(14)
print(content)
f.seek(0)
content = f.read(14)
print(content)
f.close()
