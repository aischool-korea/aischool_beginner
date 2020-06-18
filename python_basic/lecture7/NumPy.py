import numpy

# list_mat = [[1,2,3], [3,6,9], [2,4,6]]
list_mat = [[1,2,3], [4,5,6], [7,8,9]]
matrix = numpy.array(list_mat)
# print(matrix)
# print(matrix.shape)
# print(matrix[1:3, 1:3])
# print(matrix + 3)

list_mat = [[1,0,0], [0,1,0], [0,0,1]]
matrix2 = numpy.array(list_mat)

print(matrix * matrix2)
# matrix = numpy.random.rand(3,3)
# print(matrix)

# matrix = numpy.zeros((3,3))
# print(matrix)

# matrix = numpy.loadtxt("populations.txt")
# print(matrix)
# numpy.savetxt("populations2.txt", matrix)

# print(matrix[1,2])
# print(matrix[1])
# print(matrix[1:3])
#
# print(matrix+3)