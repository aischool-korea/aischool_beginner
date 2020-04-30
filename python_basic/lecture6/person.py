from abc import *
#
# class Person:
#     def __init__(self):
#       self.num_arm = 2
#     def greeting(self):
#         print('안녕하세요')
#
# class Student(Person):
#     # def __init__(self, semester):
#     #     super().__init__()
#     #     self.semester = semester
#     def study(self):
#         print('공부하기')

class StudentBase(metaclass=ABCMeta):
    @abstractmethod
    def study(self):
        pass

    @abstractmethod
    def go_to_school(self):
        pass

class Student(StudentBase):
    def study(self):
        print('공부하기')

    def go_to_school(self):
        print('학교가기')

# class Person:
#     def __init__(self):
#         self.num_arm = 2
#     def greeting(self):
#         print('안녕하세요.')
# #
# class University:
#     def credit_show(self):
#         print("A")
# #
# class Student(Person, University):
#     def __init__(self, semester):
#         super().__init__()
#         self.semester =semester
#     def greeting(self):
#         super().greeting()
#         print(f'석사과정 {self.semester}학기생입니다.')

