from builtins import range
from builtins import object
import numpy as np

from funkcije.layers import *
from funkcije.layer_utils import *


class TwoLayerNet(object):
    """
    Dvoslojna neuralna mreža sa ReLU aktivacionom funkcijom i softmaks
    funkcijom cilja koja koristi modularni dizajn sloja. Pretpostavljamo
    da je ulaz dimenzija D, H dimenzija skrivenog sloja i izvršava se 
    klasifikacija nad C klasa.
    
    Arhitektura treba da bude affine - relu - affine - softmax.

    Primijetite da ova klasa ne implementira algoritam opadajućeg gradijenta, 
    već interaguje sa odvojenim Solver objektima koji je odgovoran za 
    optimizaciju.
    
    Parametri modela koji se mogu naučiti su smješteni u rečnik 
    self.params koji mapira imena parametara u numpy nizove.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Inicijalizacija nove mreže.

        Ulazi:
        - input_dim: Veličina ulaza
        - hidden_dim: Veličina skirvenog sloja
        - num_classes: Broj klasa
        - weight_scale: Skalar koji predstavlja standardnu devijaciju
          za slučajno inicijalizovane težine.
        - reg: Snaga L2 regularizacije.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Inicijalizovati težine i bias dvoslojne mreže. Težine treba inicijalizovati    
        # Gausovom raspodjelom centriranom u 0.0 sa standardnom devijacijom weight_scale               
        # i biasima inicijalizovanim na nulu. Svi biasi i težine trebaju biti smješteni           
        # u rečnik self.params, sa prvim slojem težina i biasa koristeći ključeve 'W1' i 'b1'      
        #  i za drugi sloj težine i bias koriste ključeve 'W2' i 'b2'.                         
        ############################################################################
        pass
        ############################################################################
        #                             KRAJ KODA                                    #
        ############################################################################


    def loss(self, X, y=None):

        """
        Računanje funkcije cilja i gradijenata za minibatch podataka.

        Ulazi:
        - X: Niz ulaznih podataka oblika (N, d_1, ..., d_k)
        - y: Niz labela oblika (N,). y[i] daje labelu za X[i].

        Rezultat:
        Ako je y None, onda se pokreće test-vrijeme prolaza unaprijed modela i vraća:
        - scores: Niz dimenzija (N, C) koji daje skorove klasifikacije, gdje
          scores[i, c] je klasifikacioni skor za X[i] i klasu c.

        Ako y nije None, onda se pokreće trening-vrijeme prolaza unaprijed i unazad i 
        vraća tuple:
        - loss: Skalar koji predstavlja vrijednost funkcije cilja
        - grads: Rečnik sa istim ključevima kao self.params, mapira imena parametara
          u gradijente funkcije cilja.
        """
        scores = None
        ############################################################################
        # TODO: Implementacija prolaza unaprijed za dvoslojnu mrežu, računajuću skor
        # klase za X i smještajući ga u varijablu scores.              
        ############################################################################
        pass
        ############################################################################
        #                             KRAJ KODA                                    #
        ############################################################################

        # Ako je y None onda smo u testu i vraćamo samo scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implementacija prolaza unazad za dvoslojnu mrežu. Smjestiti funkciju cilja
        # u varijablu loss i gradijente u grads rečnik. Računanj podatke loss koristeći
        # softmax, i budi siguran da grads[k] predstavljaju gradiente za 
        # self.params[k]. Ne zaboravi dodati L2 regularizaciju.          
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    Dvoslojna neuralna mreža sa proizvoljnim brojem skrivenih slojeva,
    ReLU nelinearnošću i softmax funkcijom cilja. Ovdje će se implementirati
    dropout i batch/layer normalizacija. Za mrežu sa L slojeva, arhitektura 
    će biti

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    gdje batch/layer normalizacija i dropout su opcioni i {...} blok se
    ponavlja L - 1 puta.

    Slično dvoslojnoj mreži iznad, parametri se smještaju u
    self.params rečnik i biće naučeni koristeći Solver klasu.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Inicijalizacija nove FullyConnectedNet.

        Ulazi:
        - hidden_dims: Lista cijelih brojeva koji predstavljaju veličinu svakog skrivenog sloja.
        - input_dim: Veličina ulaza.
        - num_classes: Broj klasa.
        - dropout: Skalar između 0 i 1 koji predstavlja snagu dropout-a. Ako je dropout=1 onda
          mreža ne treba da koristi dropout.
        - normalization: Tip normalizacije koji će se koristiti. Ispravne vrijednosti
          su "batchnorm", "layernorm", ili None (default).
        - reg: Snaga L2 regularizacije.
        - weight_scale: Skalar koji predstavlja standardnu devijaciju
          za slučajno inicijalizovane težine.
        - dtype: Numpy datatype objekat; svi proračuni će biti izvršeni koristeći
          ovaj datatype. float32 je brži ali manje precizan, pa bi trebalo koristiti
          float64 za numeric gradient checking.
        - seed: Ako nije None, onda proslijeditiovaj slučajni seed dropout slojevima. Ovo
          će napraviti dropout slojeve determinističkim.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Inicijalizovati težine i bias dvoslojne mreže. Težine treba inicijalizovati    
        # Gausovom raspodjelom centriranom u 0.0 sa standardnom devijacijom weight_scale               
        # i biasima inicijalizovanim na nulu. Svi biasi i težine trebaju biti smješteni           
        # u rečnik self.params, sa prvim slojem težina i biasa koristeći ključeve 'W1' i 'b1'      
        #  i za drugi sloj težine i bias koriste ključeve 'W2' i 'b2', itd. 
        #                                                                          
        # Kada se koristi batch normalizacija, smjestiti scale i shift parametre za 
        # prvi sloj u gamma1 i beta1; za drugi sloj koristiti gamma2 i beta2 itd.     
        # Scale parametri trebaju biti inicijalizovani na jedinice i shift     
        # parameetri trebaju biti inicijalizovani na nule.                               
        ############################################################################
        pass
        ############################################################################
        #                             KRAJ KODA                                    #
        ############################################################################

        # Kada se koristi dropout potrebno je proslijediti dropout_param rečnik svakom
        # dropout sloju tako da sloj zna dropout vjerovatnoću i mod (train / test). 
        # Možeš proslijediti isti ropout_param svakom dropout sloju.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # Sa batch normalizacijom potrebno je voditi računa o trenutnim srednjim vrijednostima
        # i varijansama, pa je potrebno proslijediti bn_param objekat svakom batch normalizacionom
        # sloju. Trebaš proslijediti self.bn_params[0] prolazu unaprijed prvog batch
        # normalizacionog sloja, self.bn_params[1] prolazu unaprijed drugom sloju itd.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Računanje funkcije cilja i gradijenata za FC mrežu.

        Ulaz / izlaz: Isto kao kod TwoLayerNet iznad.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Podesi train/test mode za batchnorm params i dropout param jer se ponašaju
        # drugačije tokomtreniranja i testiranja.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implementacija prolaza unaprijed za dvoslojnu mrežu, računajuću skor
        # klase za X i smještajući ga u varijablu scores.          
        #                                                                          
        # Kada se koristi dropout, proslijedi self.dropout_param svakom       
        # dropout forward pass.                                                    
        #                                                                          
        # Kada se koristi batch normalization, proslijedi self.bn_params[0]  
        # prolazu unaprijed za prvi batch normalizacioni sloj, proslijedi           
        # self.bn_params[1] za drugi batch normalizacioni sloj, itd.                                                             
        ############################################################################
        pass
        ############################################################################
        #                             KRAJ KODA                                    #
        ############################################################################

        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implementacija prolaza unazad za dvoslojnu mrežu. Smjestiti funkciju cilja
        # u varijablu loss i gradijente u grads rečnik. Računanj podatke loss koristeći
        # softmax, i budi siguran da grads[k] predstavljaju gradiente za 
        # self.params[k]. Ne zaboravi dodati L2 regularizaciju.              
        #                                                                          
        # Kada se koristi batch/layer normalizacija, nije potrebno regularizovati skalu
        # i pomjerati parametre.                                                    
        ############################################################################
        pass
        ############################################################################
        #                             KRAJ KODA                                    #
        ############################################################################

        return loss, grads
