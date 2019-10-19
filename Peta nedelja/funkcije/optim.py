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


def adam(x, dx, config=None):
    """
    Realizuje Adam.

    config format:
    - learning_rate: Skalar stopa učenja.
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
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)

    next_x = None
    beta1, beta2, eps = config['beta1'], config['beta2'], config['epsilon']
    t, m, v = config['t'], config['m'], config['v']
    m = beta1 * m + (1 - beta1) * dx
    v = beta2 * v + (1 - beta2) * (dx * dx)
    t += 1
    alpha = config['learning_rate'] * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    x -= alpha * (m / (np.sqrt(v) + eps))
    config['t'] = t
    config['m'] = m
    config['v'] = v
    next_x = x

    return next_x, config
