import numpy as np

f = open('tail.txt','r')
g = open('tail2.txt','w')

for line in f:
    table = line.split()
    kappa_greco = float(table[4])/(2.*2.*2.*2.) * (float(table[3])/float(table[1]))**5
    g.write('%s \n' %(kappa_greco))

f.close()
g.close()


