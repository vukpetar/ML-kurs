from funkcije.layers import *
from funkcije.fast_layers import *


def conv_relu_forward(x, w, b, conv_param):
    """
    Realizuje sloj koji vrši konvoluciju i ima ReLU aktivacionu funkciju.

    Argumenti:
    - x: Ulaz u konvolucioni sloj
    - w, b, conv_param: Težine i biasi konvolucionog sloja

    Rezultat je tuple koji sadrži:
    - out: Izlaz iz ReLU-a
    - cache: Vrijednosti koje očekuje prolaz unazad
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Prolaz unazad za conv-relu sloj.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Realizuje sloj koji vrši konvoluciju, ima ReLU aktivacionu funkciju i vrši pool operaciju.

    Argumenti:
    - x: Ulaz u konvolucioni sloj
    - w, b, conv_param: ežine i biasi konvolucionog sloja
    - pool_param: Parametri pooling sloja

    Rezultat je tuple koji sadrži:
    - out: Izlaz iz pooling sloja
    - cache: rijednosti koje očekuje prolaz unazad
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Prolaz unazad za conv-relu-pool sloj
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
