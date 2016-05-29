import random

f = open("data_banknote_authentication.txt", 'r')

readlines = f.readlines()
random.shuffle(readlines)

f.close()

f = open("data_banknote_authentication_rand.txt", 'w')

for i in readlines:
    f.write(i)

f.close()
