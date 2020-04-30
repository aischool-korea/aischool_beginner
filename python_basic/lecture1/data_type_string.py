
print("Hello World")
print('Hello World')

print("a")
print('123', end=' ')
print('456')


c = "Kim's paper"
print(c)
#
d = 'He said "Hi"'
print(d)

c = 'Kim\'s paper'
print(c)
#
d = "He said \"Hi\""
print(d)

multiline = "Life is too short\nYou need python"
print(multiline)

multiline = """
Life is too short
You need python
Kim's paper
"""
print(multiline)
# #
multiline = '''
Life is too short
You need python
'''
print(multiline)

teacher = "Kim's "
title = "AI School"
print(teacher + title)
#
print("="*30)

print(teacher + title)
print("="*30)

print(len(title))
print(title[0])
print(title[1])
print(title[-1])
print(title[0:3])
print(title[3:])

odd_even = "홀짝홀짝홀짝"
print(odd_even[::2])
print(odd_even[1::2])

a = "apple"
print(a.count("p"))

print(a.find("c"))
print(a.index("c"))

print(" ".join((a)))

a = a.upper()
print(a)
print(a.lower())

num = "234"
print(num)
print(type(num))
num = float(num)
print(num)

num = str(num)
print(num)

b = " How can I improve my coding skills? \n"
print(b)

b = b.strip()
print(b)

b = b.replace("?", "!")
print(b)
word_list = b.split(" ")
print(word_list)

apple_num = 3
orange_num = 2
apple_num_string = "three"
print("I eat %d apples." % apple_num)
print("I eat {0} apples.".format(apple_num))
print("I eat %s apples." % apple_num_string)
print("I eat %d apples and %d oranges." % (apple_num, orange_num))
print("I eat {0} apples and {1} oranges.".format(apple_num, orange_num))
print("I eat {apple_num} apples and {orange_num} oranges.".format(apple_num=1, orange_num=2))

print("Error is %d%%." % 98)

print(f'I eat {apple_num} apples and {orange_num} oranges.')

pi = 3.141592
print("pi = %f" % pi)
print("pi = %0.4f" % pi)