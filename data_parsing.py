#import os

#os.chdir('/Users/minhyeok/Desktop/rudska')    #사용하면 파일안에 있는 모든 자료를 한번에 할 수 있습니다.

with open("155.txt", 'r') as file:    #파일을 열어서 새로 csv파일로 깔끔하게 정리해서 만듭니다.
    file2 = open("dateasdasdsad.csv", 'w+')

    lines = file.readlines()

    for line in lines:
        line = line.rstrip()[10:]
        file2.write(line + '\n')
