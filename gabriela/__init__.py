from gabriela.gabrielaGPU import DNN
from gabriela.gabrielaGPU import set_activate_func, get_activate_func
from gabriela.gabrielaGPU import FUNC_ID_TANH, FUNC_ID_SIGMOID, FUNC_ID_RELU, FUNC_ID_ALAN

# for retro compatible version
RND = DNN

if __name__ == '__main__':  # teste

    input = [[1, 1], [1, 0], [0, 1], [0, 0]]
    target = [[int(i ^ j)] for i, j in input]
    dnn = DNN((2, 6, 5, 3, 1))
    epocas = 1000
    error = 1e-3
    set_activate_func(FUNC_ID_ALAN)
    for ep in range(epocas):
        e = 0
        for i in range(len(input)):
            dnn(input[i])
            dnn.aprender(target[i])
            e += (dnn.out[0] - target[i][0]) ** 2
        e /= 2
        if e < error:
            print('after ', ep, 'epics i learn it, erro =', e)
            for i in range(len(input)):
                x, y = input[i]
                print(x, 'xor', y, '=', dnn(input[i]))
            dnn.aprender(target[i])
            break;

'''
    # set_activate_func(FUNC_ID_ALAN)
    from time import time
    t0 = time()
    a = DNN((2, 1400,800,220,28, 1))
    dt = time()-t0
    print('criar rede ',dt,'s')

    inp = [[1, 1], [1, 0], [0, 1], [0, 0]]
    out = [[int(k and v)] for k,v in inp]
    _ = 0
    # print(a.showc())
    f = open('testnew.txt', 'w')
    t0 = time()
    epocas = 10000
    for _ in range(epocas):
        for i in range(len(inp)):
            # print(_ + 1, a(inp[0]), file=f)
            a(inp[i])
            a.aprender(out[i])
        # print(a([1,1]))
        if (a([1,1] )[0]>0.999):
            break;
    dt = time()-t0
    print(f'{_} epocas tempo',dt,'s')
f.close();
# a.save('redes/g1.rng')
print(_ + 1, a([1, 1]))
a = None
# b = RND.load('redes/g1.rng')
# print(_ + 1, b([1, 1]))
'''
