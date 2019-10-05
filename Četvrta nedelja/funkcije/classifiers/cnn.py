from builtins import object
import numpy as np

from funkcije.layers import *
from funkcije.fast_layers import *
from funkcije.layer_utils import *


class ThreeLayerConvNet(object):
    """
    Troslojna konvoluciona mreža sledeće arhitekture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    Mreža radi na minibatches podataka koji imaju dimenzije (N, C, H, W)
    sastoje se od N slika, svaka visne H i širine W i sa C ulaznih kanala
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Inicijalizacija nove mreže.

        Ulazi:
        - input_dim: Tuple (C, H, W) veličina ulaznog podatka
        - num_filters: Broj filtera koji će se koristiti u konvolucionom sloju
        - filter_size: Širina/visina filtra u konvolucionom sloju
        - hidden_dim: Broj ćelija u FC skrivenom sloju
        - num_classes: Broj skorova koje treba dobiti u poslednjem affine sloju.
        - weight_scale: Standarna devijacija za slučajnu inicijalizaciju težina.
        - reg: Snaga L2 reguularizacije
        - dtype: numpy tip podatka
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Inicijalizujte te-ine i biase za troslojnu konvolucionu mrežu. Težine 
        # treba inicijalizovati Gausovom raspodjelom, a biase na nulu. Sve težine i
        # biase treba upisati u rečnik self.params koristeći ključeve 'W1' i 'b1'
        # za konvolucioni sloj; ključevi 'W2' i 'b2' za skriveni affine sloj;
        # i ključevi 'W3' i 'b3' za izlazni affine sloj.                           
        ############################################################################
        pass
        ############################################################################
        #                             KRAJ KODA
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluacija funkcije cilja i gradijenata za troslojnu konvolucionu mrežu.

        Ulaz / izlaz: Isti kao što smo imali kod dvoslojne mreže kod FC zadatka.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # proslijediti conv_param prolazu unaprijed za konvolucioni sloj
        # Proširivanje i pomjeranje da bi se očuvale prostorne veličine
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # proslijediti pool_param prolazu unaprijed za max-pooling sloj
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implementirajte prolaz unaprijed za troslojnu konvolucionu mrežu, 
        # računanjem skorova klasa za X i smještajući ih u varijablu scores.
        #                                                                          
        # Napomena: Možete koristiti funkcije definisane u funkcije/fast_layers.py i  
        # funkcije/layer_utils.py u vašoj implementaciji (već realizovane).         
        ############################################################################
        pass
        ############################################################################
        #                             KRAJ KODA                             
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implementirajte prolaz unazad za troslojnu konvolucionu mrežu
        # smještajući loss i gradijente u varijable loss i grads. Izračunajte funkciju cilja 
        # koristeći softmax. Ne zaboravite dodati L2 regularizaciju.
        #                                                                          
        # Napomena: Da bi bili sigurni da se vaša implementacija poklapa sa našom
        # i dobijemo željenu grešku, provjerite vašu L2 regularizaciju tako da uključi faktor
        # 0.5 da bi pojednostavili izraz za gradijent.
        ############################################################################
        pass
        ############################################################################
        #                             KRAJ KODA
        ############################################################################

        return loss, grads
