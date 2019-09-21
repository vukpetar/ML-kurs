import numpy as np
from random import randrange

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    Naivna implementacija numeričkog gradijenta funkcije
    f u x.
    - f treba da bude funkcija koja ima jedan argument
    - x je tačka (numpy niz) u kojoj treba evaluirati gradijent
    """

    fx = f(x) # evaluacija vrijednosti funkcije u originalnoj tački
    grad = np.zeros_like(x)
    # iteracija kroz sve indekse u x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluacija funkcije u x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h 
        fxph = f(x) 
        x[ix] = oldval - h
        fxmh = f(x) 
        x[ix] = oldval 

        # računanje parcijalnih izvoda pomoću centrirane formule
        grad[ix] = (fxph - fxmh) / (2 * h) # nagib
        if verbose:
            print(ix, grad[ix])
        it.iternext() # korak ka sledećoj dimenziji

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluacija numeričkog gradijenta za funkciju koja prima numpy niz
    i kao rezultat vraća numpy niz.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
    Računanje numeričkih gradijenata za funkciju koja kao ulaz i
    izlaz ima blobove.

    Pretpostavka je da funkcija f prima nekoliko ulaznih blobova kao argument, 
    praćeni izlaznim blobom gdje će izlazi biti upisani. Na primjer,
    funkcija f se može pozvati kao:

    f(x, w, out)

    gdje su x i w ulazni blobovi, a rezultat funkcije f će biti upisan u out.

    Ulazi:
    - f: funkcija
    - inputs: tuple ulaznih blobova
    - output: izlazni blob
    - h: veličina koraka
    """
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals, flags=['multi_index'],
                       op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    return eval_numerical_gradient_blobs(lambda *args: net.forward(),
                inputs, output, h=h)


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    Odabiranje nekoliko slučajnih elemenata i vraćanje samo numeričke u ovim dimenzijama.
    """

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h 
        fxph = f(x)
        x[ix] = oldval - h 
        fxmh = f(x) 
        x[ix] = oldval

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = (abs(grad_numerical - grad_analytic) /
                    (abs(grad_numerical) + abs(grad_analytic)))
        print('numerički: %f analitički: %f, relativna greška: %e'
              %(grad_numerical, grad_analytic, rel_error))
