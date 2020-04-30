def sum(a, b):
    return a + b

print(sum(3, 5))
print(sum(2, 1))

def sum_and_mul(a, b):
    return a + b, a*b

s, m = sum_and_mul(3,5)
print(s)
print(m)
print(sum_and_mul(3, 5))
print(sum_and_mul(2, 1))