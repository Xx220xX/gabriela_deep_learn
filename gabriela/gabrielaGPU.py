# https://medium.com/binaryandmore/beginners-guide-to-deriving-and-implementing-backpropagation-e3c1a5a1e536
from gabriela.gabrielac import *
import random

random.seed(1)


def rand():
    return random.random() * 2 - 1


def newRandomList(length):
    return [rand() for i in range(length)]


class DNN:#deep neural network
    def __init__(self, arquitetura, taxaAprendizado=0.1):
        self.gab = c_GAB()
        self.L = len(arquitetura) - 1
        self.n = arquitetura
        self.p_inp = c.c_double * self.n[0]  # to fast create input
        self.p_out = c.c_double * self.n[-1]
        m = c_Mat * (self.L + 1)
        self.gab.a = m(newMati(self.n[0], 1, 1))
        self.gab.w = m()
        self.gab.b = m()
        self.gab.z = m()
        # self.gab.a
        for i in range(1, self.L + 1):
            self.gab.w[i] = newMat(self.n[i], self.n[i - 1], rand)  # weight nl x nl+1
            self.gab.b[i] = newMat(self.n[i], 1, rand)  # bias nl x 1
            self.gab.a[i] = newMat_empty(self.n[i], 1)  # activate vector nlx1
            self.gab.z[i] = newMat_empty(self.n[i], 1)  # sum vector nlx1

        self.hitlearn = taxaAprendizado
        self.gab.arq_0 = c.c_int(self.n[0])
        self.gab.arq_o = c.c_int(self.n[-1])
        self.gab.L = c.c_int(self.L)

    def __call__(self, input):
        temp = self.p_inp(*input)
        clib.call(c.addressof(self.gab), temp)
        self.out = [self.gab.a[self.L].v[i] for i in range(self.n[-1])]
        return self.out

    def aprender(self, true_output):
        temp = self.p_out(*true_output)
        clib.aprende(c.addressof(self.gab), temp, c.c_double(self.hitlearn))

    def save(self, file2save):
        vet = c.c_int * len(self.n)
        return clib.save(c.c_char_p(file2save.encode('utf-8')), c.addressof(self.gab), c.c_int(len(self.n)),
                         vet(*self.n))

    @staticmethod
    def load(file2load):
        c_arq = c_vector()
        check = clib.preLoad(c.c_char_p(file2load.encode('utf-8')), c.addressof(c_arq))
        if int(check) != 0:
            raise FileNotFoundError('arquivo nao encontrado')
        arq = [c_arq.p[i] for i in range(c_arq.len)]
        rede = DNN(arq)
        check = clib.load(c.c_char_p(file2load.encode('utf-8')), c.addressof(c_arq), c.addressof(rede.gab))
        if int(check) != 0:
            raise FileNotFoundError('arquivo nao encontrado')
        return rede
