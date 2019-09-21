import numpy as np


def affine_forward(x, w, b):
    """
    Izračunava prolaz unaprijed za afina (FC) sloj.

    Ulaz x ima dimenzije (N, d_1, ..., d_k) i sadrži mini batch of N
    primjera, gdje svaki primjer x[i] ima dimenzije (d_1, ..., d_k). Moramo
    predimenzionisati svaki ulaz u vektor dimenzija D = d_1 * ... * d_k, a onda
    da dobijemo izlaz dimenzija M.
    
    Argumenti:
    - x: numpy array koji sadrži ulazne podatke, dimenzija (N, d_1, ..., d_k)
    - w: numpy array koji sadrži težine, dimenzija (D, M)
    - b: numpy array koji sadrži biase, dimenzija (M,)

    Rezultat je tuple koji sadrži:
    - out: izlaz, dimenzija (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implementirati afina prolaz unaprijed. Smjestiti rezultat u out.  #
    # Ulaz se mora predimenzionisati u redove.                                #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    """
    Izračunava prolaz unazad za afina (FC) sloj.

    Argumenti:
    - dout: Izvod koji dolazi "odozgo", dimenzija (N, M)
    - cache: Tuple koji sadrži:
      - x: Ulazni podaci, dimenzija (N, d_1, ... d_k)
      - w: Težine, dimenzija (D, M)
      - b: Biasi, dimenzija (M,)

    Rezultat je tuple koji sadrži:
    - dx: Gradijent po x, dimenzija (N, d1, ..., d_k)
    - dw: Gradijent po w, dimenzija (D, M)
    - db: Gradijent po b, dimenzija (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implementirati afina prolaz unazad.                               #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Izračunava prolaz unaprijed za ReLU.

    Argumenti:
    - x: Ulaz, bilo kojih dimenzija

    Rezultat je tuple koji sadrži:
    - out: Izlaz, istih dimenzija kao x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implementirati ReLU prolaz unaprijed.                             #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Izračunava prolaz unazad za ReLU.
    
    Argumenti:
    - dout: Izvod koji dolazi "odozgo", bilo kojih dimenzija
    - cache: Ulaz x, istih dimenzija kao dout

    Rezultat sadrži:
    - dx: Gradijent po x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implementirati ReLU prolaz unazad.                                #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Prolazak unaprijed za batch normalizaciju.

    Tokom treninga srednja vrijednost i varijansa računaju se po mini batch-u
    i koriste se za normalizaciju statistike ulaznih podataka. Takođe, tokom treninga
    pamtimo eksponencijalno opadajuću srednju vrijednost i varijansu koja će se
    za vrijeme testiranja koristiti. 
    
    Ažuriranje tih promjenljivih vrši se na sljedeći način:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Rad za batch normalizaciju predlaže drugi pristup u kojem se za svaku osobinu
    ulaznih podataka računa running_mean i running_var što mi nećemo raditi 
    kako bi uštedjeli na vremenu izvršivanja.
  

    Argumenti:
    - x: Podaci dimenzija (N, D)
    - gamma: Parametar skaliranja dimenzija (D,)
    - beta: Shift parametar dimenzija (D,)
    - bn_param: Dictionary sa ključevima:
      - mode: 'train' ili 'test'; obavezan
      - eps: Konstanta za numeričku stabilnost
      - momentum: Konstanta za running mean / variance.
      - running_mean: Array dimenzija (D,) 
      - running_var Array dimenzija (D,)

    Rezultat je tuple koji sadrži:
    - out: izlaz dimenzija (N, D)
    - cache: Tuple vrijednosti potrebnih za korak unazad
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype = x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype = x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implementirati training-time prolaz unaprijed za batch norm.  #
        # Upotrebom minibatch statistike izračunajte srednju vr. i varijansu, #
        # i iskoristite te vrijednosti da normalizujete ulazne podatke i      #
        # izvrište njihovo pomjeranje upotrebom gamma i beta parametara       #
        #                                                                     #
        # U promjenljivoj out izračunajte izlaz. Svaka promjenljiva koja vam  #
        # je potrebna u koraku unazad treba da se nađe u cache promjenljivoj  #
        #                                                                     #
        # Bez obzira što se računa running_var, za normalizaciju se koristi   #
        # standardna devijacija odnosno korijen varijanse.                    #
        # Detaljnije na (https://arxiv.org/abs/1502.03167)                    #
        #######################################################################
        pass
        #######################################################################
        #                            VAŠ KOD SE OVDJE ZAVRŠAVA                #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implementirati test-time prolaz unaprijed za batch norm.      #
        # Iskoristite running_mean i running_var da norm. test podatke.       #
        # Nakon toga skalirajte i šiftujte podatke upotrebom gamma i beta.    #
        # Rezultat upišite u promjenljivoj out.                                #
        #######################################################################
        pass
        #######################################################################
        #                            VAŠ KOD SE OVDJE ZAVRŠAVA                #
        #######################################################################
    else:
        raise ValueError('Pogrešan režim rada: "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Prolazak unazad za batch normalizaciju.

    Prvo je potrebno na papiru izvesti backpropagation za batch normalizaciju,
    a onda ga realizovati u ovoj funkciji.

    Argumenti:
    - dout: Izvod koji dolazi "odozgo", dimenzija N, D)
    - cache: Promjenljive iz batchnorm_forward.

    Rezultat je tuple koji sadrži:
    - dx: Gradijent po  x, dimenzija (N, D)
    - dgamma: Gradijent po gamma, dimenzija (D,)
    - dbeta: Gradijent po beta, dimenzija (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implementirati prolaz unazad za batch normalizaciju. Upisati rez. #
    # u dx, dgamma, i dbeta promjenljivim.                                    #
    # Osvrnuti se na rad (https://arxiv.org/abs/1502.03167)                   #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternativno rješenje za prolazak unazad kod batch normalizacije.

    Za implementaciju ovog zadatka potrebno je opet izvesti izvode na papiru
    imajući u vidu da treba koristiti što manje koraka. Obratiti pažnju na 
    postavku.
     
    Napomena: Ova realizacija očekuje istu cache promjenljivu kao i prethodna,
    ali možda neće koristiti sve od njih.

    Argumenti / rezultat: Isti kao i u prethodnoj funkciji
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implementacija bi trebalo da bude u jednoj liniji od 80 karaktera #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Prolaz unaprijed za sloj normalizaciju.

    Tokom treninga i testa, ulazni podaci su normalizovani po svakom ulaznom podatku,
    prije nego što su skalirani sa gamma i beta parametara što je identično batch normali-
    zaciji.

    Ponašanje u ova dva režima je isto u odnosu na batch normalizaciju, ali nemamo running_mean
    i running_var promjenljive. 

    Argumenti:
    - x: Podaci dimenzija (N, D)
    - gamma: Parametar skaliranja dimenzija (D,)
    - beta: Parametar šiftovanja dimenzija (D,)
    - ln_param: Dictionary sa sledećim ključevima:
        - eps: Konstanta za numeričku stabilnost

    Rezultat je tuple koji sadrži:
    - out: dimenzija (N, D)
    - cache: Tuple promjenljivih potrebnih za prolaz unazad
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implementirati training-time prolaz unaprijed za sloj norm.       #
    # Napomena: ovo se može uraditi malom izmjenom funkcije za batch norm.    #
    # Smislite kako.                                                          #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Prolaz unazad za sloj normalizaciju.

    Ponovo se možete osloniti na funkciju koja je već realizovana odnosno batch
    normalizaciju.

    Argumenti:
    - dout: Izvod koji dolazi "odozgo", dimenzija N, D)
    - cache: Promjenljive iz layernorm_forward.

    Rezultat je tuple koji sadrži:
    - dx: Gradijent po x, dimenzija (N, D)
    - dgamma: Gradijent po gamma, dimenzija (D,)
    - dbeta: Gradient po beta, dimenzija (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implementirati prolaz unazad za normalizaciju po sloju            #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Prolaz unaprijed za invertovani dropout.

    Argumenti:
    - x: Ulazni podaci bilo kojih dimenzija
    - dropout_param: Dictionary sa sledećim ključevima:
      - p: Dropout parametar. Ostavljamo svaki neuron sa vjerovatnoćom p.
      - mode: 'test' ili 'train'. Ako je režim train, vršimo dropout;
        ako je režim test onda preskačemo ovu operaciju.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Rezultat:
    - out: Array istih dimenzija kao x.
    - cache: tuple (dropout_param, mask). U trening režimu maska je maska kojom
      se množi sa ulazom, a u test režimu maska je None.

    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implementirati trening fazu i sačuvati masku u promjenljivoj  #
        # mask                                                                #
        #######################################################################
        pass
        #######################################################################
        #                            VAŠ KOD SE OVDJE ZAVRŠAVA                #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO:  Implementirati test fazu za invertovani dropout              #
        #######################################################################
        pass
        #######################################################################
        #                            VAŠ KOD SE OVDJE ZAVRŠAVA                #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy = False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Prolaz unazad za invertovani dropout.

    Argumenti:
    - dout: Izvod koji dolazi "odozgo", bilo kojih dimenzija
    - cache: (dropout_param, mask) iz dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implementirati trening fazu prolaska unazad za dropout        #
        #######################################################################
        pass
        #######################################################################
        #                            VAŠ KOD SE OVDJE ZAVRŠAVA                #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


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
