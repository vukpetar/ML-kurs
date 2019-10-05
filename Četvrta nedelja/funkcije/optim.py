import numpy as np

"""
U ovom fajlu vrši se realizacija nekoliko načina ažuriranja gradijenata. Svaka od
funkcija uzima trenutne vrijednosti težina i gradijent loss-a po tim težinama,
a kao rezultat daje ažurirane težine. Svako pravilo za ažuriranje ima sledeći 
interfejs:

def update(w, dw, config=None):

Argumenti:
  - w: numpy array sa trenutnim težinama.
  - dw: numpy array istih dimenzija kao w koji predstavlja izvod loss-a po w.
  - config: Dictionary koji sadrži hiperparametre kao što su learning rante, momentum...
    Ukoliko su cache vrijednosti potrebne, config će ih sadržati.

Rezultat:
  - next_w: Sledeća vrijednost težina
  - config: Koji se prosleđuje sledećoj iteraciji.

NOTE: Za većinu ovih funkcija podrazumijevani learning rate obično neće zadovoljavati;
ostali hiperparametri vjerovatno hoće.
"""


def sgd(w, dw, config = None):
    """
    Realizuje vanila SGD.

    config format:
    - learning_rate: Skalar stopa učenja.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Realizuje SGD sa momentuom.
    
    config format:
    - learning_rate: Skalar stopa učenja.
    - momentum: Skalar između 0 i 1.
      Postavljanje momentum = 0 je vanila sgd.
    - velocity: numpy array istih dimenzija kao w i dw u kojem se smješta pokretna
      sredina gradijenata. 
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implementirati moment ažuriranje formulu. Ažurirane vrijednosti   #
    # složiti u next_w. Treba koristiti i ažurirati v.                        #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################
    config['velocity'] = v

    return next_w, config



def rmsprop(w, dw, config=None):
    """
    Realizuje RMSProp.

    config format:
    - learning_rate: Skalar stopa učenja.
    - decay_rate: Skalar između 0 i 1 koji predstavlja koeficijent zaboravljanja. 
    - epsilon: Mali skalar kojim se izbjegava dijeljenje sa nulom.
    - cache: Pokretna sredina drugih momenata gradijenta.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implenetirati formulu za RMSProp, smještajuću sledeću vrijednost  #
    # w u next_w. Ne zaboravite da ažurirate nove vrijednosti cache-a u       #
    # config['cache'].                                                        #
    ###########################################################################
    pass
    ###########################################################################
    #                            VAŠ KOD SE OVDJE ZAVRŠAVA                    #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Realizuje Adam.

    config format:
    - learning_rate: SSkalar stopa učenja.
    - beta1: Koeficijent zaboravljanja za prvi moment gradijenta.
    - beta2: Koeficijent zaboravljanja za drugi moment gradijenta.
    - epsilon: Mali skalar kojim se izbjegava dijeljenje sa nulom.
    - m: pokretna sredina gradijenta.
    - v: pokretna sredina kvadriranog gradijenta.
    - t: broj iteracije.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None
    ###########################################################################
    # TODO: Implementirati formulu za Adam smještajući sledeću vrijednost w   #
    # u next_w. Ne zaboravite da ažurirate m, v, and t promjenljive           #
    # smještene u config.                                                     #
    #                                                                         #
    # NOTE: Prvo ažurirajte t prije bilo kakvih kalkulacija                   #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config
