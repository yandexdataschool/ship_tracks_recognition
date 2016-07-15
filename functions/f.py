__author__ = 'mikhail91'



def f(a, b):
    return a*b

from multiprocessing import Pool


def ff():
    p = Pool(2)
    return p.map(f, [1, 2, 3])
