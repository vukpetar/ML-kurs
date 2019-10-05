import numpy as np

def conv_forward_naive(x, w, b, conv_param):
    """
    Naivna implementacija prolaza unaprijed konvolucionog sloja.

    Ulaz se sastoji od N ulaza, svaki ima C kanala, visinu H i širinu W.
    Konvoluiramo svaki od ulaza sa F filtera gdje svaki filter filtrira svih
    C kanala ulaznog podatka i ima visinu HH i širinu WW.
    
    Argumenti:
    - x: Ulazni podaci dimenzija (N, C, H, W)
    - w: Težine filtra dimenzija (F, C, HH, WW)
    - b: Bias, dimenzija (F,)
    - conv_param: Dictionary sa sledećim ključevima:
      - 'stride': Korak pomjeranja u oba prava, horizontalno i vertikalno. 
      - 'pad': Broj piksela kojim će se izvršiti zero-padding ulaza.
        
    Tokom operacije dodavanja nula, 'pad' nule se trebaju dodati simetrično po visinu i 
    širini ulaza. Ulazni podatak se ne smije izmijeniti.

    Rezultat je tuple koji sadrži:
    - out: Izlazni podaci, dimenzija (N, F, H', W') gdje su H' i W' dobijeni kao
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implementirajte konvolucioni prolaz unaprijed.                    #
    # Hint: np.pad je funkcija kojom se mogu dodati nule.                     #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    Naivna implementacija prolaza unazad konvolucionog sloja.

    Argumenti:
    - dout: Izvod koji dolazi "odozgo"
    - cache: Tuple koji sadrži (x, w, b, conv_param) iz conv_forward_naive

    Rezultat je tuple koji sadrži:
    - dx: Gradijent po x
    - dw: Gradijent po w
    - db: Gradijent po b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implementirajte konvolucioni prolaz unazad.                       #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    Naivna implementacija max-pooling sloja za prolaz unaprijed.

    Argumenti:
    - x: Ulazni podaci, dimenzija (N, C, H, W)
    - pool_param: Dictionary sa sledećim ključevima:
      - 'pool_height': Visina svake pooling regije
      - 'pool_width': Širina svake pooling regije
      - 'stride': Korak pomijeranja

    Dodavanje nula nije potrebno. 

    Rezultat je tuple koji sadrži:
    - out: Izlazni podaci, dimenzija (N, C, H', W') gdje su H' i W' dobijeni kao
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implementirajte max-pooling prolaz unaprijed.                     #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    Naivna implementacija max-pooling sloja za prolaz unazad.
    
    Argumenti:
    - dout: Izvod koji dolazi "odozgo"
    - cache: Tuple koji sadrži (x, pool_param) iz max_pool_forward_naive

    Rezultat:
    - dx: Gradijent po x
    """
    dx = None
    ###########################################################################
    # TODO: Implementirajte max-pooling prolaz unazad.                        #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Izračunava prolaz unaprijed za prostornu batch normalizaciju.
    
    Argumenti:
    - x: Ulazni podaci dimenzija (N, C, H, W)
    - gamma: Parametar skaliranja, dimenzija (C,)
    - beta: Parametar šiftovanja, dimenzija (C,)
    - bn_param: Dictionary sa sledećim ključevima:
      - mode: 'train' ili 'test'; obavezan
      - eps: Konstanta za numeričku stabilnost
      - momentum: Konstanta za running mean / variance. momentum = 0 znači da
        se stara informacija u potpunosti odbacije u svakom prolazu, dok
        momentum = 1 znači da se nova informacija nikad ne uključuje. 
        Podrazumijevana vrijednost momentum = 0.9 bi trebalo da radi u većini slučajeva
      - running_mean: Array dimenzija (D,) daje running mean feature-a
      - running_var Array dimenzija (D,) daje running variance feature-a

    Rezultat je tuple koji sadrži:
    - out: Izlazni podaci, dimenzija (N, C, H, W)
    - cache: Vrijednosti potrebne za prolaz unazad
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implementirajte prolaz unaprijed za prostornu BN.                 #
    #                                                                         #
    # HINT: Možete implementirati prostornu BN upotrebom koda za BN koji ste  #
    # implementirati prethodne sedmice. Rješenje bi trebalo da sadrži oko     #
    # 5 linija koda.                                                          #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Izračunava prolaz unazad za prostornu batch normalizaciju.

    Argumenti:
    - dout: Izvod koji dolazi "odozgo", dimenzija (N, C, H, W)
    - cache: Vrijednosti iz prolaza unaprijed

    Rezultat je tuple koji sadrži:
    - dx: Gradijent po ulazu, dimenzija (N, C, H, W)
    - dgamma: Gradijent po parametru skaliranja, dimenzija (C,)
    - dbeta: Gradijent po parametru šiftovanja,  dimenzija (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implementirajte prolaz unazad za prostornu BN.                    #
    #                                                                         #
    # HINT: Možete implementirati prostornu BN upotrebom koda za BN koji ste  #
    # implementirati prethodne sedmice. Rješenje bi trebalo da sadrži oko     #
    # 5 linija koda.                                                          #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Izračunava prolaz unaprijed za prostornu grupnu normalizaciju.

    Inputs:
    - x: Ulazni podaci dimenzija (N, C, H, W)
    - gamma: Parametar skaliranja, dimenzija (C,)
    - beta: Parametar šiftovanja, dimenzija (C,)
    - G: Cio broj koji predstavlja broj grupa, djelilac broja C
    - gn_param: Dictionary sa sledećim ključevima:
    - eps: Konstanta za numeričku stabilnost

    Rezultat je tuple koji sadrži:
    - out: Izlazni podaci, dimenzija (N, C, H, W)
    - cache: Vrijednosti potrebne za prolaz unazad
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implementirajte prolaz unaprijed za prostornu GN. Imati na umu da $
    # implementacija dosta liči na sloj normalizaciju (Layer Normalization)   #
    # pa se dobar dio koda može upotrebiti.                                   #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Izračunava prolaz unazad za prostornu grupnu normalizaciju.

    Inputs:
    - dout: Izvod koji dolazi "odozgo", dimenzija (N, C, H, W)
    - cache: Vrijednosti iz prolaza unaprijed

    Rezultat je tuple koji sadrži:
    - dx: Gradijent po ulazu, dimenzija (N, C, H, W)
    - dgamma: Gradijent po parametru skaliranja, dimenzija (C,)
    - dbeta: Gradijent po parametru šiftovanja, dimenzija (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implementirajte prolaz unazad za prostornu GN                     #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################
    return dx, dgamma, dbeta

def svm_loss(x, y):
    """
    Računa loss i gradijent za višeklasni SVM.

    Argumenti:
    - x: Ulazni podaci, dimenzija (N, C) gdje je x[i, j] score za j-tu klasu
      i-tog ulaza.
    - y: vektor labela, dimenzija (N,) gdje je y[i] labela za x[i] i
      0 <= y[i] < C

    Rezultat je tuple koji sadrži:
    - loss: Skalar koji nam daje loss
    - dx: Gradijent loss-a po x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis = 1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Računa loss i gradijent za višeklasni softmax klasifikaciju.

    Argumenti:
    - x: Ulazni podaci, dimenzija (N, C) gdje je x[i, j] score za j-tu klasu
      i-tog ulaza.
    - y: vektor labela, dimenzija (N,) gdje je y[i] labela za x[i] i
      0 <= y[i] < C

    Rezultat je tuple koji sadrži:
    - loss: Skalar koji nam daje loss
    - dx: Gradijent loss-a po x
    """
    shifted_logits = x - np.max(x, axis = 1, keepdims = True)
    Z = np.sum(np.exp(shifted_logits), axis = 1, keepdims = True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
