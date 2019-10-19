from __future__ import print_function, division
from builtins import range
import numpy as np


"""
U ovom fajlu su definisani tipovi slojeva koji se često koriste za rekurentne neuralne mreže.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Prolaz unaprijed za jedan vremenski korak vanila RNN koji koristi tanh 
    aktivacionu funkciju.

    Ulazni podatak je dimenzija D, skriveno stanje je dimenzije H i koristimo 
    minibatch veličine N.

    Ulazi:
    - x: Ulazni podatak za ovaj vremenski korak, dimenzija (N, D)
    - prev_h: Skriveno stanje iz prethodnog vremenskog koraka, dimenzija (N, H)
    - Wx: Matrica težina za ulaz-skrivene konekcije, dimenzija (D, H)
    - Wh: Matrica težina za skrivene-skrivene konekcije, dimenzija (H, H)
    - b: Biasi dimenzija (H,)

    Vraća tuple:
    - next_h: Sledeće skriveno stanje, dimenzija (N, H)
    - cache: Tuple vrijednosti potrebnih za prolaz unazad
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implementirajte jedan korak prolaza unaprijed za vanila RNN. Smjestite sledeće  
    # skriveno stanje i svaku potrebnu vrijednost za računanje prolaza unazad u next_h   
    # i cache varijable respektivno.                                          
    ##############################################################################
    

    pass

    
    ##############################################################################
    #                               KRAJ KODA                                    
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Prolaz unazad za jedan vremenski korak vanila RNN.

    Ulazi:
    - dnext_h: Gradijent funkcije cilja po sledećem skrivenom stanju, dimenzija (N, H)
    - cache: Cache objekat iz prolaza unaprijed

    Vraća tuple:
    - dx: Gradijenata ulaznih podataka, dimenzija (N, D)
    - dprev_h: Gradijenata prethodnog skrivenog stanja, dimenzija (N, H)
    - dWx: Gradijenata težina ulaz-skriveni, dimenzija (D, H)
    - dWh: Gradijenata težina skriveni-skriveni, dimenzija (H, H)
    - db: Gradijenata bias vektora, dimenzija (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implementirajte prolaz unazad za jedan korak vanila RNN.      
    #                                                                            
    # POMOĆ: Za tanh funkciju, možete da izračunate lokalni izvod u smislu
    # izlazne vrijednosti iz tanh.                                             
    ##############################################################################
    

    pass

    
    ##############################################################################
    #                               KRAJ KODA                             
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Vanila RNN prolaz unaprijed na cijeloj sekvenci podatak. Pretpostavljamo da je ulaz
    sekvenca od T vektora, svaki dimenzija D. RNN koristi skriveni dimenzija H i radimo na
    minibatch koji sadrži N sekvenci. Nakon pokretanja prolaza unaprijed, vraćamo
    skriveno stanje svih vremenskih koraka.

    Ulazi:
    - x: Ulazni podatak za cijeli vremenski interval, dimenzija (N, T, D)
    - h0: Početno skriveno stanje, dimenzija (N, H)
    - Wx: Matrica težina ulaz-skriveni konekcija, dimenzija (D, H)
    - Wh: Matrica težina skrivena-skrivena konekcija, dimenzija (H, H)
    - b: Biasi dimenzija (H,)

    Vraća tuple:
    - h: Skrivenih stanja za cijeli vremenski interval, dimenzija (N, T, H)
    - cache: Vrijednosti potrebne za prolaz unazad
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implementacija prolaza unaprijed za vanila RNN na sekvenci ulaznih podataka.
    # Trebate koristiti rnn_step_forward funkciju koju ste definisali iznad.
    # Možete koristiti for petlju da bi vam pomogla da izračunate prolaz unaprijed.       
    ##############################################################################
    

    pass

    
    ##############################################################################
    #                               KRAJ KODA
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Računanje prolaza unazada za vanila RNN nad cijelom sekvencom podataka.

    Ulazi:
    - dh: Gradijenti svih skrivenih stanja, dimenzija (N, T, H) 
    
    NAPOMENA: 'dh' sadrži gradijente proizvedene od stane individualnih
    funkcija cilja u svakom vremenskom koraku, *ne* gradijenti 
    između vremenskih koraka (koje ćete morati izračunati pozivajući
    rnn_step_backward u petlji).

    Vraća tuple:
    - dx: Graijenata ulaza, dimenzija (N, T, D)
    - dh0: Gradijenata početnog skrivenog stanja, dimenzija (N, H)
    - dWx: Gradijenata težina ulaz-skriveni, dimenzija (D, H)
    - dWh: Gradijenata težina skriveni-skriveni, dimenzija (H, H)
    - db: Gradijenata biasa, dimenzija (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implementacija prolaza unazad za vanila RNN na cijeloj sekvenci podataka.
    # Trebate koristiti rnn_step_backward funkciju koju ste definisali iznad.
    # Možete koristiti for petlju za računanje prolaza unazad.   
    ##############################################################################


    pass


    ##############################################################################
    #                               KRAJ KODA
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Prolaz unaprijed za word embeddings. Radimo na minibatch veličine N gdje svaka 
    sekvenca ima dužinu T. Pretpostavljamo da vokabular od V riječi, dodjeljuje svakoj 
    riječi vektor dužine D.

    Ulazi:
    - x: Integer niz dimenzija (N, T) koji predstavlja indekse riječi. Svaki element idx
      od x mora biti u opsegu 0 <= idx < V.
    - W: Matrica težina dimenzija (V, D) koja predstavlja vektore riječi za sve riječi.

    Vraća tuple:
    - out: Niza dimenzija (N, T, D) koji predstavlja vektore riječi za sve ulazne riječi.
    - cache: Vrijednosti potrebne za prolaz unazad.
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implementirajte prolaz unaprijed za word embeddings.                      
    #                                                                            
    # POMOĆ: Može biti odrađeno u jednoj liniji koristeći NumPy-ovo indeksiranje nizova.           
    ##############################################################################
    

    pass

    
    ##############################################################################
    #                               KRAJ KODA                             
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Prolaz unazad za word embeddings. Ne možemo back-propagate u riječima s obzirom
    da su cijeli brojevi, tako da vraćamo gradijente za matricu word embedding.

    Ulazi:
    - dout: Gradijenti dimenzija (N, T, D)
    - cache: Vrijednosti iz prolaza unaprijed

    Vraća:
    - dW: Gradijent matrice word embedding, dimenzija (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implementacija prolaza unazad za word embeddings.                     
    #                                                                            
    # Napomena: Riječ se može pojaviti više od jednom u sekvenci.
    # POMOĆ: Pogledajte funkciju np.add.at                                       
    ##############################################################################
    

    pass

    
    ##############################################################################
    #                               KRAJ KODA                                    #
    ##############################################################################
    return dW


def sigmoid(x):
    
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Prolaz unaprijed za jedan vremenski korak LSTM-a.

    Ulazni podatak je dimenzija D, skriveno stanje H i koristimo minibatch veličine N.

    Ulazi:
    - x: Ulazni podatak, dimenzija (N, D)
    - prev_h: Prethodno skriveno stanje, dimenzija (N, H)
    - prev_c: Prethodno stanje ćelije, dimenzija (N, H)
    - Wx: Težine ulaz-skriveni, dimenzija (D, 4H)
    - Wh: Težine skriveni-skriveni, dimenzija (H, 4H)
    - b: Biasi, dimenzija (4H,)

    Vraća tuple:
    - next_h: Sledeće skriveno stanje, dimenzija (N, H)
    - next_c: Sledeće stanje ćelije, dimenzija (N, H)
    - cache: Tuple vrijednosti potrebnih za prolaz unazad.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implementirajte prolaz unaprijed za jedan vremenski korak LSTM-a. Možete
    # koristiti datu verziju sigmoida iznad.
    #############################################################################

    pass

    #
    ##############################################################################
    #                               KRAJ KODA                                    #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Prolaz unazad za jedan vremenski korak LSTM-a.

    Ulazi:
    - dnext_h: Gradijenti sledećeg skrivenog stanja, dimenzija (N, H)
    - dnext_c: Gradijenti sledećeg stanja ćelije, dimenzija (N, H)
    - cache: Vrijednosti iz prolaza unaprijed

    Vraća tuple:
    - dx: Gradijenat ulaznih podataka, dimenzija (N, D)
    - dprev_h: Gradijent prethodnog stanja, dimenzija (N, H)
    - dprev_c: Gradijent prethodnog stanja ćelije, dimenzija (N, H)
    - dWx: Gradijent težina ulaz-skriveni, dimenzija (D, 4H)
    - dWh: Gradijent težina skriveni-skriveni, dimenzija (H, 4H)
    - db: Gradijent biasa, dimenzija (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implementirajte prolaz unazad za jedan vremenski korak LSTM-a.
    # POMOĆ: Za sigmoid i tanh možete računati lokalne izvode u pogledu izlazne 
    # vrijednosti iz nelinearnosti.                                 
    #############################################################################

    pass

    #
    ##############################################################################
    #                               KRAJ KODA                                    #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Prolaz unaprijed za LSTM nad cijelom sekvencom podataka. Pretpostavljamo da je
    ulaz sekvenca od T vektora, svaki dimenzije D. LSTM koristi skrivene veličine 
    H i radimo sa minibatch-em koji sadrži N sekvenci. Nakon pokretanja prolaza unaprijed,
    vraćamo skrivena stanja za sve vremenske korake.

    Primijetite da je početno stanje ćelije proslijeđeno kao ulaz, ali početno stanje
    ćelije je podešeno na nulu. Takođe primijetite da se stanje ćelije ne vraća kao izlaz;
    to je interna varijabla za LSTM i ne pristupa joj se izvan.

    Ulazi:
    - x: Ulazni podatak dimenzija (N, T, D)
    - h0: Početno skriveno stanje dimenzija (N, H)
    - Wx: Težine veza ulaz-skriveni, dimenzija (D, 4H)
    - Wh: Težine veza skriveni-skriveni, dimenzija (H, 4H)
    - b: Biasi dimenzija (4H,)

    Vraća tuple:
    - h: Skrivena stanja za sve vremenske korake svih sekvenci, dimenzija (N, T, H)
    - cache: Vrijednosti za prolaz unazad
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implementacija prolaza unaprijed za LSTM nad cijelim vremenskim     #
    # intervalom. Trebate koristiti funkciju lstm_step_forward function koju    #
    # ste realizovali iznad.                                                    #
    #############################################################################

    pass

    #
    ##############################################################################
    #                               KRAJ KODA                                    #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Prolaz unazad kroz LSTM ćeliju po cijeloj sekvenci podataka.

    Ulazi:
    - dh: Gradijenti skrivenih stanja, dimenzija (N, T, H)
    - cache: Vrijednosti iz prolaza unaprijed

    Vraća tuple:
    - dx: Gradijanata po ulaznim podacima, dimenzija (N, T, D)
    - dh0: Gradijenti po skrivenim stanjima, dimenzija (N, H)
    - dWx: Gradijenti po parametrima od ulaza do skrivenih stanja, dimenzija (D, 4H)
    - dWh: Gradijenti parametara od skrivenih do skrivenih tanja, dimenzija (H, 4H)
    - db: Gradijenti bias-a, dimenzija (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implementirajte prolaz unazad kroz LSTM ćeliju po Cijeloj sekvenci  #
    # podataka.Treba da koristite funkciju lstm_step_backward definisanu iznad. #
    #############################################################################
    #

    pass

    #
    ##############################################################################
    #                               KRAJ KODA                                    #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Prolaz naprijed kroz vremenski afajn sloj. Ulaz je skup D-dimenzionih vektora
    poređanih u minibatch-ove od N vremenskih serija, gdje svaka serija je dužine T.
    Uloga ovog sloja je da ulazne vektore transformiše u nove vektore veličine M.

    Ulazi:
    - x: Ulazni podaci dimenzija (N, T, D)
    - w: Težine dimenzija (D, M)
    - b: Bias-i dimenzija (M,)

    Vraća tuple:
    - out: Izlazni podaci dimenzija (N, T, M)
    - cache: Vrijednosti potrebne za prolaz unazad
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Prolaz unazad kroz vremenski afajn sloj.

    Input:
    - dout: Gradijenti prethodnog sloja dimenzija (N, T, M)
    - cache: Vrijednosti iz prolaza naprijed

    Returns a tuple of:
    - dx: Gradijenti po x-u, dimenzija (N, T, D)
    - dw: Gradijenti po težinama, dimenzija (D, M)
    - db: Gradijenti po bias-ima, dimenzija (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    Vremenska verzija softmax funkcije gubitka koja se koristi u RNN mrežama. 
    Predpostavka je da se predikcija pravi nad vokabularom koji je dužine V 
    u svakom vremenskom trenutku t. Sekvenca koja pokušava da se predvidi je
    dužine T i to se radi nad minibatch-om veličine N. Ulaz x daje skorove za 
    svaki element u vokabularu za sve vremenske trenutke. Koristićemo cross-entropy
    funkciju gubitka u svakom vremenskom trenutku, sumiraćemo ih po svim vremenskim 
    trenucima i usrednjiti po minibatch-u.

    Dodatna komplikacija je ako imamo elemente u minibatch-u koji imaju različite veličine
    sekvenci, u tom slučaju ćemo da napravimo masku za svaki element u minibatc-u kako bi
    uticaj na funkciju gubitka imali samo elementi koji se nalaze u okvirima dužine sekvence
    za određeni element u minibatch-u.

    Ulazi:
    - x: Skorovi, dimenzija (N, T, V)
    - y: Labele, dimenzija (N, T) gdje je svaki element u opsegu
         0 <= y[i, t] < V
    - mask: Niz Boolean-a dimenzija (N, T) gdje mask[i, t] nam govori da li x[i, t] treba da 
    utiče na funkciju gubitka ili ne.

    Vraća tuple:
    - loss: Skalar koji predstavlja funkciju gubitka
    - dx: Izvod funkcije gubitka u odnosu na x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
