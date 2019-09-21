from funkcije.layers import *

def affine_relu_forward(x, w, b):
    """
    Realizuje prolaz unaprijed sloja koji ima afina transformaciju i ReLU aktivaciju.

    Argumenti:
    - x: Ulaz u afina sloj
    - w, b: Težina za afina sloj

    Rezultat je tuple koji sadrži:
    - out: Izlaz iz ReLU
    - cache: Objekat za prolazak unazad
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Realizuje prolaz unazad sloja koji ima afina transformaciju i ReLU aktivaciju.
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db