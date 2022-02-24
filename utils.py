import defs
import numpy as np

def mprint(matrix):
    for row in matrix:
        for val in row:
            print(defs.fprint_format % val, end='')
        print()