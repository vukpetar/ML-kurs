from __future__ import print_function, division
from builtins import range
from builtins import object
import numpy as np

from funkcije import optim
from funkcije.coco_utils import sample_coco_minibatch


class CaptioningSolver(object):
    """
    CaptioningSolver objedinjuje svu logiku potrebnu da bi se trenirao
    model za objašnjavanje slika. Klasa CaptioningSolver koristi algoritam
    stohastičkog gradijentnog spuštanja sa različitim pravilima ažuriranja 
    parametara koja su defnisisana u optim.py fajlu.

    CaptioningSolver kao argumente prihvata podatke za obuku i podatke za
    validaciju kako bi mogao periodično da provjerava tačnost klasifikatora
    kako na skupu podataka za obuku tako i na skupu za validaciju a sve to
    ima za cilj da se izbjegne pretreniranje.

    Da bi istrenirali model potrebno je prvo da kreirate instancu klase
    CaptioningSolver, kojoj treba u konstruktoru da proslijedite model, 
    skup podataka, različite opcione parametre (stopa učenja, veličina batch-a i sl.).
    Nakon toga potrebno je da pokrenete metodu train() kako bi započeli reniranje
    modela.

    Kada se završi treniranje modela, kao rezultat funkcije train() dobijamo
    model.params - parametri mreže koji imaju najveću tačnost na validacionom 
    skupu podataka.
    Instanca klase CaptionSolver takođe sadrži: solver.loss_history - lista
    koja sadrži vrijednosti funkcije cilja tokom procesa treniranja,
    solver.train_acc_history - lista koja sadrži tačnosti na skupu podataka
    za obuku tokom treniranja, solver.val_acc_history - lista koja sadrži 
    tačnosti na skupu podataka za validaciju tokom treniranja

    Primjer kako se može kreirati instanca CaptioningSolver klase:

    data = load_coco_data()
    model = MyAwesomeModel(hidden_dim=100)
    solver = CaptioningSolver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()

    CaptioningSolver klasa radi sa objektom modela koji mora da
    zadovoljava sledeća pravila:
    - model.params mora da bude rečnik čiji su ključevi imena
    parametara, a vrijednosti numpy nizovi koji predstavljaju 
    vrijednosti parametara
    - model.loss(features, captions) mora da bude funkcija koja
    računa funkciju cilja i gradijente tokom treninga sa sledećim
    ulazima i izlazima:

    Ulazi:
     - features: Niz koji predstavlja minibath karakteristika za slike,
     dimenzija (N, D)
     - captions: Niz objašnjenja za odgovarajuće slike iz niza features,
     dimenzija (N, T) gdje svaki element je u osegu od (0, V]

     Vraća:
     - loss: Skalar koji predstavlja funkciju cilja
     - grads: Rečnik sa istim ključevima kao i self.params samo
     što su vrijednosti gradijenti funkcije cilja u odnosu na
     te parametre
    """

    def __init__(self, model, data, **kwargs):
        """
        Konstruktor klase CaptioningSolver.

        Obavezni argumenti:
        - model: Instanca modela koja zadovoljava gore navedena pravila
        - data: Rečnik koji sadrži podatke za obuku i validaciju koji se
        dobija iz funkcije load_coco_data

        Opcioni argumenti:
        - update_rule: String koji predstavlja naziv algoritma za ažuriranje
        parametara (u optim.py fajlu).
          Podrazumijevani je 'sgd'.
        - optim_config: Rječnik koji sadrži hiperparametre koji će biti proslijeđeni 
        odabranom pravilu ažuriranja. Svako pravilo ažuriranja zahtijeva različite 
        hiperparametre (pogledajte optim.py), ali sva pravila ažuriranja zahtijevaju parametar
        'learning_rate'.
        - lr_decay: Skalar za opadanje brzine učenja; nakon svake epohe stopa 
        učenja se množi s ovom vrijednošću.
        - batch_size: Veličina minibatch-a koja se koristi za računanje funkcije
        gubitka i gradijenta tokom treninga.
        - num_epochs: Broj epoha.
        - print_every: Integer; funkcija gubitka će biti štampana svakih print_every
        iteracija.
        - verbose: Boolean; indikator da li će funkcija gubitka biti štampana tokom
        treninga.
        """
        self.model = model
        self.data = data

        # Otpakivanje keyword argumenata
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Baca grešku ako ima višak keyword argumenata
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Nepoznati argumenti %s' % extra)

        # Provjera da li pravilo ažuriranja parametara postoji
        # i nakon toga promjena stringa sa stvarnom funkcijom koja
        # vrši ažuriranje parametara
        if not hasattr(optim, self.update_rule):
            raise ValueError('Nevažeći update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()


    def _reset(self):
        """
        Inicijalizacija nekih varijabli za optimizaciju. Ova funkcija 
        ne smije da se poziva ručno.
        """
        # Inicijalizacija varijabli
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Pravljenje kopije parametara optim_config varijable
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d


    def _step(self):
        """
        Ažuriranje parametara. Ova funkcija se poziva metodom train()
        i ne bi trebala da bude pozivana manuelno.
        """
        # Pravljenje minibatch-a podataka za obuku
        minibatch = sample_coco_minibatch(self.data,
                      batch_size=self.batch_size,
                      split='train')
        captions, features, urls = minibatch

        # Računanje funkcije gubitka i gradijenata
        loss, grads = self.model.loss(features, captions)
        self.loss_history.append(loss)

        # Ažuriranje parametara
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config


    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Provjera tačnosti modela na podacima koji su dati funkciji.

        Ulazi:
        - X: Niz podataka, dimenzija (N, d_1, ..., d_k)
        - y: Niz labela, dimenzija (N,)
        - num_samples: Ako nije None, testira se model na podskupu podataka veličine num_samples.
        - batch_size: X i y se dijele u batch-eve veličine batch_size kako bi se
        izbjeglo korištenje previše memorije.

        Vraća:
        - acc: Skalar koji predstavlja odnos pravilno klasifikovanih primjeraka i ukupnog
        broja primjeraka koje je klasifikovao model.
        """
        return 0.0

        # Kreiranje podskupa podataka ako je definisan parametar num_samples
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Računanje predikcija po batch-evima
        num_batches = N / batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc


    def train(self):
        """
        Pokretanje optimizacije za računanje modela.
        """
        num_train = self.data['train_captions'].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Štampanje vrijednosti funkcije gubitka za skup podataka za obuku,
            # ako je parametar verbose = True
            if self.verbose and t % self.print_every == 0:
                print('(Iteracija %d / %d) Vrijednost funkcije gubitka: %f' % (
                       t + 1, num_iterations, self.loss_history[-1]))

            # Na kraju svake epohe, povećava se brojač (self.epoch) i smanjuje se stopa učenja
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

