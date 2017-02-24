import matplotlib.pyplot as plt
##
smallSet = [(100, 520),(500, 2865),(1000, 6029),(2000, 12858),(3000,19725),(4000,27367),(5000,35133),(8000,57567),(10000,74731),(16000,121741),(28000,225184),(32000,271714),(48000,419607),(64000,564580)]

X, Y = [a for (a, _) in smallSet], [b/1000 for (_, b) in smallSet]

plt.plot(X, Y, 'b-')
plt.plot(X, Y, 'ro')
##
from math import *

X=[0.01*i for i in range(0, 100)]
Y=[i*i*i*i*i for i in X]
plt.plot(X, Y)
plt.show()
##
file = open("D:\\Projets\\Map generator\\tests.txt", 'r')
temps = [int(t) for t in file.readlines()]
effectifs=[0] * (max(temps)+1)
for t in temps:
    effectifs[t] += 1
hauteurs=[effectifs[t] for t in temps]
plt.bar(temps, hauteurs)
