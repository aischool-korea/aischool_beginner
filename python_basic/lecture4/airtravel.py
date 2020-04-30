class Flight:
    nation = 'Korea'
    def __init__(self, number, passenger_num):
        # print('init')
        # if not number[:2].isalpha():
        #     raise ValueError("첫 두글자가 알파벳이 아닙니다.")
        # if not number[:2].isupper():
        #     raise ValueError("첫 두글자가 대문자가 아닙니다.")
        # if not number[2:].isdigit():
        #     raise ValueError("세번째 글자 이상이 양의 숫자가 아닙니다.")
        # self.__number = number

        self.__number = number
        self._passenger_num = passenger_num

    # def __new__(cls):
    #     print('new')
    #     return super().__new__(cls)

    def number(self): #메소드 작성
        return self.__number
    #
    def add_passenger(self, num):
        self._passenger_num += num