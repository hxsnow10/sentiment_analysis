import multiprocessing as mp
import itertools
import time


def g():
    for el in xrange(50):
        print el
        yield el


def f(x):
    time.sleep(1)
    return x * x

if __name__ == '__main__':
    pool = mp.Pool(processes=4)              # start 4 worker processes
    go = g()
    result = []
    N = 11
    while True:
        g2 = pool.map(f, itertools.islice(go, N))
        if g2:
            result.extend(g2)
            time.sleep(1)
        else:
            break
    print(result)
