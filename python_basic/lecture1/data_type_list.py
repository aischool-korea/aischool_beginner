a = [1, 2, 3, 4, 5]
print(a)
print(len(a))

b = ["a", "b", "c", "d"]
print(b)
print(len(b))

c = [1, "a", 2.5]
print(c)
print(len(c))


a = [1, 2, 3, 4, 5]
print(a)
print(len(a))
print(a[0])
print(a[1])
print(a[-1])
print(a[0:2])
print(a[:2])
print(a[2:])
print(a+b)
print(a*3)
print(len(a))

d = [1, 2, [3, 4, 5]]
print(d)
print(len(d))
print(d[2][:2])
print(d[0][2])

a = [1, 2, 3, 4, 5]
a[2] = 6
print(a)
print(a.index(4))
del a[2]
print(a)
del a[2:]
print(a)
a.append(2)
print(a)
print(a.count(2))
print(max(a))
print(min(a))
print(sum(a))


a = [2, 1, 5, 4, 3]
b = sorted(a)
print(a)
print(b)
a.sort()
print(a)

a.reverse()
print(a)
a.insert(0, 6)
print(a)
a.remove(6)
print(a)
print(a.pop())
print(a)
print(a.pop())
print(a)
a.extend([1, 0])
print(a)


a = ["b", "a", "d", "c"]
a.sort()
print(a)

numbers = [1, 2, 3, 4, 5]
result = []
for n in numbers:
    if n % 2 == 1:
        result.append(n*2)
print(result)

result = [n*2 for n in numbers if n % 2 == 1]
print(result)
