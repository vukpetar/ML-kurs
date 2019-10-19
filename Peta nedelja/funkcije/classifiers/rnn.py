from builtins import range
from builtins import object
import numpy as np

from funkcije.layers import *
from funkcije.rnn_layers import *


class CaptioningRNN(object):
    """
    Klasa CaptioningRNN proizvodi opise iz karakteristika slika koristeći rekurentnu
    neuralnu mrežu.

    RNN kao ulaz prima vektore dimenzija D, ima vokabular veličine V, radi sa 
    sekvencama dužine T, ima skriveni dimenzije H, koristi vektore riječi 
    dimenzija W i radi na minibatch-evima veličine N.

    Napomena: Ne koristimo regularizaciju.
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
        """
        Konstrukcija nove CaptioningRNN instance

        Ulazi:
        - word_to_idx: Rečnik daje vokabular. Sadrži V riječi i
          mapira svaki string jedinstvenim cijelim brojem u opsegu [0, V).
        - input_dim: Dimenzije D vektora ulazne karakteristike slike.
        - wordvec_dim: Dimenzije W vektora riječi.
        - hidden_dim: Dimenzije H za skriveno stanje RNN-a.
        - cell_type: Tip ćelije RNN-; 'rnn' ili 'lstm'.
        - dtype: numpy tip podatka koji se koristi; koristite float32 za trening i float64 
          za provjeru gradijenata.
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Inicijalizacija vektora riječi
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # Inicijalizacija CNN -> parametri projekcije skrivenog stanja
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Inicijalizacija parametara za RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Inicijalizacija izlaza na težine vokabulara
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, features, captions):
        """
        Računanje funkcije cilja za RNN. Ulaz su karakteristike slike i pravi
        opisi za te slike, i koristimo RNN (ili LSTM) da izračunamo funkciju
        cilja i gradijente svih parametara.

        Ulazi:
        - features: Ulazne karakteristike slike, dimenzija (N, D)
        - captions: Pravi opisi; Niz cijelih brojeva dimenzija (N, T) gdje
          je svaki element u opsegu 0 <= y[i, t] < V

        Vraća tuple:
        - loss: Vrijednost funkcije cilja
        - grads: Rečnik gradijenata paralelno self.params
        """
        # Podjela opisa u dva dijela: captions_in ima sve osim poslednje riječi
        # i to je ulaz u RNN; captions_out ima sve osim prve riječi. 
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        mask = (captions_out != self._null)

        # Težina i bias za afajn transformaciju iz karakteristika slike u početno skriveno stanje.
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrica
        W_embed = self.params['W_embed']

        # Ulaz-skriveni, skriveni-skriveni, i biasi za RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Težina i bias za skriveni-vokabular transformaciju
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implementirajte prolaze unaprijed i unazad za CaptioningRNN.
        # U prolazu unaprijed ćete morati da odradite sledeće:                   
        # (1) Koristite afajn transformaciju da biste izračunali početno skriveno
        #     stanje iz karakteristika slike. Kao rezultat dobićete niz dimenzija (N, H)
        # (2) Koristite word embedding sloj da transformišete riječi u captions_in od
        #     indeksa do vektora, rezultirajući u niz dimenzija (N, T, W).         
        # (3) Koristite vanila RNN ili LSTM (zavisi od self.cell_type) da obradite
        #     sekvencu vektora ulaznih riječi i dobijete vektore skrivenog stanja za
        #     sve vremenske korake, dimenzija (N, T, H).   
        # (4) Koristite afajn transformaciju da izračunate scores
        #     za vokabula u svakom vremenskom koraku koristeći skrivena stana, dimenzija (N, T, V).
        # (5) Koristite softmax za računanje funkcije cilja koristeći captions_out, ignorišujući
        #     trenutke gdje je izlaz <NULL> koristeći masku iznad.     
        #                                                                          
        # U prolazu unazad računaćete gradijente u odnosu na sve parametre.                                              
        ############################################################################
        

        pass

        
        ############################################################################
        #                             KRAJ KODA
        ############################################################################

        return loss, grads


    def sample(self, features, max_length=30):
        """
        Pokreće prolaz unaprijed za testiranje modela.

        U svakom vremenskom korak, embed trenutnu riječ, proslijedi je i prethodno
        skriveno stanje RNN dobija sledeće skriveno stanje, koristi skriveno stanje da 
        se dobiju scores za sve riječi vokabulara i bira riječ sa najboljim score kao
        sledeću riječ. Početno skriveno stanje se računa primjenjujući afajn transformaciju
        ulaznim karakteristikama slike i početna riječ je <START> token.

        Za LSTM ćete takođe pratiti stanje ćelije; u tom slučaju početno stanje ćelije
        bi trebalo da bude nula.

        Ulazi:
        - features: Niz ulaznih karakteristika slike dimenzija (N, D).
        - max_length: Maksimalna dužina T generisanih opisa.

        Vraća:
        - captions: Niz dimenzija (N, max_length) dajući opise, gdje je svaki element
          cijeli broj u opsegu [0, V). Prvi element opisa bi trebalo da bude riječ, ne
          <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        ###########################################################################
        # TODO: Trebate inicijalizovati skriveno stanje RNN-a primjenjujući afajn transformaciju
        # na ulazne karakteristike slike. Prva riječ je <START> token; njena vrijednost
        # je smještena u varijablu self._start. U svakom vremenskom koraku trebate da:          
        # (1) Embed prethodnu riječ koristeći naučene word embeddings           
        # (2) Napravite RNN korak koristeći prethodno skriveno stanje i embed trenutnu riječ
        #     da biste dobili sledeće skriveno stanje.                          
        # (3) Primijenite naučenu afajn transformaciju na sledeće skriveno stanje
        #     da biste dobili scores za sve riječi u vokabularu.                  
        # (4) Odaberete riječ sa najvećim scores kao sledeću riječ, upišete je 
        #     (indeks riječi) na odgovarajuće mjesto u varijabli captions. 
        #                                                                         
        # Radi jednostavnosti, ne morate da prestanete generisanje nakon <END> tokena, 
        # ali možete ako želite.                                 
        #                                                                         
        # POMOĆ: Nećete moći da koristite funkcije rnn_forward ili lstm_forward;
        # već ćete pozivati funkcije call rnn_step_forward ili lstm_step_forward u petlji.                                                                 
        #                                                                         
        # NAPOMENA: Još uvijek radimo na minibatch-evima u ovoj funkciji. Ako radite sa LSTM
        # podesite početno stanje na nulu.        
        ###########################################################################


        pass

        
        ############################################################################
        #                             KRAJ KODA
        ############################################################################
        return captions
