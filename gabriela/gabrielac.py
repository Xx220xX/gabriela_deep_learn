import ctypes as c
import os,sys
from platform import architecture

FUNC_ID_TANH = 0
FUNC_ID_RELU = 2
FUNC_ID_SIGMOID = 8
FUNC_ID_ALAN = 16

clib = None


if architecture()[0] == '64bit':
    dir = os.path.dirname(sys.modules['gabriela'].__file__)
    temp = os.path.join(dir, "lib/gabriela64.dll")
    
    clib = c.CDLL(temp)
    # clib = c.OleDLL(temp)
else:
    dir = os.path.dirname(sys.modules['gabriela'].__file__)
    temp = os.path.join(dir, "lib/gabriela32.dll")
    clib = c.CDLL(temp)
clib.call.argtypes = [c.c_void_p, c.c_void_p]
clib.aprende.argtypes = [c.c_void_p, c.c_void_p, c.c_double]
clib.setFuncActivation.argtypes = [c.c_int]


class c_Mat(c.Structure):
    _fields_ = [("m", c.c_int), ("n", c.c_int), ("v", c.POINTER(c.c_double))]

    def __repr__(self):
        s = f'cMatrix {self.m}x{self.n}'
        for i in range(self.m):
            for j in range(self.n):
                s = s + f'{self.v[i * self.n + j]} '
            s = s + '\n'
        return s

    def __str__(self):
        return self.__repr__()


def newMat(m, n, f):
    v = c.c_double * (m * n)
    v = v(*[f() for i in range(m * n)])
    return c_Mat(m, n, v)


def newMati(m, n, value):
    v = c.c_double * (m * n)
    v = v(*([value] * (m * n)))
    return c_Mat(m, n, v)


def newMat_empty(m, n):
    v = c.c_double * (m * n)
    v = v()
    return c_Mat(m, n, v)


class c_vector(c.Structure):
    _fields_ = ("len", c.c_int), ("p", c.POINTER(c.c_int))


class c_GAB(c.Structure):
    _fields_ = (
        ("L", c.c_int), ("arq_0", c.c_int), ("arq_o", c.c_int), ("a", c.POINTER(c_Mat)), ("z", c.POINTER(c_Mat)),
        ("w", c.POINTER(c_Mat)), ("b", c.POINTER(c_Mat)))


def set_activate_func(FUNC) -> bool:
    "return true if sucess"
    return clib.setFuncActivation(FUNC) == 0


def get_activate_func() -> str:
    id = int(clib.getFuncActivation())
    if id == FUNC_ID_ALAN: return 'func alan'
    if id == FUNC_ID_RELU: return 'func relu'
    if id == FUNC_ID_SIGMOID: return 'func sigmoid'
    if id == FUNC_ID_TANH: return 'func tanh'
