from __future__ import print_function, division
# from future import standard_library
# standard_library.install_aliases()
from builtins import range
from builtins import object
import os
import pickle as pickle

import numpy as np

from funkcije import optim


class Solver(object):
    """
    Solver klasa obuhvata svu logiku potrebnu za treniranje modela za klasifikaciju. Solver izvršava
    algoritam stohastičkog gradijentnog spuštanja koristeći različite pravila ažuriranja
    parametara definisana u fajlu u optim.py.

    Konstruktor klase Solver kao argument ima podatke za trening, podatke za validaciju i labele
    tako da može periodično provjeravati tačnost klasifikacije i na trening i na validacijonom
    skupu podataka kako bi imali kontrolu da ne dođe do pretreniranja.

    Da bi se istrenirao model, prvo ćete konstruisati Solver instancu, tako što ćete u konstruktoru
    predati model, skup podataka i ostale parametre kao što su: brzina učenja, veličina serije itd.
    Tada ćete pozvati metodu train() kako bi započeli optimizacionu proceduru i treniranje modela.

    Nakon što se metoda train() završi, u varijabli model.params će biti sadržani parametri modela
    koji su dali najbolje rezultate na validacionom setu tokom treninga. Pored toga, varijabla
    solver.loss_history će sadržati istoriju svih gubitaka koji su se pojavili tokom treniranja
    modela, a varijabla solver.train_acc_history i solver.val_acc_history će sadržati tačnosti
    modela tokom treninga i tokom validacije tokom svake epohe.

    Primjer kako se koristi klasa Solver možete vidjeti ispod:

    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'y_val': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


   Solver klasa radi sa model objektom koji mora biti u skladu sa sljedećim API-jem:
   - model.params mora biti rečnik koji kao ključeve ima imena parametara, a kao vrijednosti numpy nizove koji predstavljaju vrijednosti parametara
   - model.loss (X, y) mora biti funkcija koja računa funkciju gubitka i gradijente tokom treninga, i klasifikacioni skor tokom testiranja
      Ulazi:
      - X: Niz koji predstavlja minibatch ulaznih podataka oblika (N, d_1, ..., d_k)
      - y: Niz labela oblika (N,) koji daju labele za X gdje je y[i]
        labela za X[i].
      Izlazi:
      Ako je y None, to znači da je faza testiranja:
      - scores: Niz oblika (N, C) dajući ocjene klasifikacije za X gdje
        vrijednost [i, c] daju vjerovatnoću klase c za X[i].

      Ako y nije None, to znači da je trening faza i potrebno je pokrenuti funkcije za prolaz unaprijed i prolaz unazad:
      - loss: Skalar koji ima vrijednost funkcije opadanja
      - grad: Rječnik sa istim ključevima kao i varijabla self.params samo što sadrži vrijednosti gradijenata za određene parametre
    """

    def __init__(self, model, data, **kwargs):
        """
        Kreirajte novu instancu klase Solver.

        Traženi argumenti:
        - model: Objekt modela u skladu s gore opisanim API-jem
        - data: Riječnik podataka za trening i validaciju koji sadrže:
          'X_train': Niz, oblika (N_train, d_1, ..., d_k) slika iz trening skupa
          'X_val': Niz, oblika (N_val, d_1, ..., d_k) slika iz validacionog skupa
          'y_train': Niz, oblika (N_train,) labela za trening slike
          'y_val': Niz, oblika (N_val,) labela za validacione slike

        Opcioni argumenti:
        - update_rule: string koji sadrži ime pravila za ažuriranja parametara u optim.py.
          Default-na vrijednost je 'sgd'.
        - optim_config: Rječnik koji sadrži hiperparametre koji će biti
          prmijenjeni na izabrano pravilo ažuriranja parametara. Svako pravilo ažuriranja zahtijeva drugačije
          hiperparametara (vidi optim.py), ali sva pravila ažuriranja zahtijevaju
          parametar 'learning_rate' tako da uvijek mora biti prisutan.
        - lr_decay: skalar za opadanje brzine učenja; nakon svake epohe
          stopa učenja množi se sa ovom vrijednošću.
        - batch_size: Veličina batch-a koje se koriste za izračunavanje funkcije gubitaka i gradijenta
          tokom treninga.
        - num_epochs: Broj epoha koje treba pokrenuti tokom treninga.
        - print_every: Integer; gubici na treningu biće ispisani svaki
          print_every iteracija.
        - verbose: Boolean; ako je postavljeno na false, tada se neće ispisati nikakav izlaz
          tokom treninga.
        - num_train_samples: Broj uzoraka treninga koji se koriste za provjeru treninga
          tačnost; default-na vrijednost je 1000; ako postavite na None onda se koristi cijeli trening skup.
        - num_val_samples: Broj uzoraka za validaciju koji se koriste tokom provjere
          tačnost; default je None, to yna;i da sekoristi cijeli validacioni skup.
        - checkpoint_name: Ako nije None, ovdje čuvate model posle svake epohe.
        """
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        # Učitavanje dodatnih argumenata
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', None)

        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Baciti grešku ako postoje dodatni argumenti
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        # Budite sigurni da pravilo za ažuriranje parametara postoji, a onda
        # zamijenite ime sa stvarnom funkcijom
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()


    def _reset(self):
        """
        Postavljanje vrijedosti promjenljivih za optimizaciju. Nemojte da zovete ovu funkciju
        nakon inicijalizacije klase.
        """
        # Postvaljanje vrijednosti varijablama
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Napraviti kopiju svih optim_config parametara
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d


    def _step(self):
        """
        Pravi jedan korak gradijentnog spuštanja. Ova funkcija se poziva iz
        funkcije train() i ne treba biti pozivana samostalno.
        """
        # Pravi minibatch od trening podataka
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Računanje funkcije gubitka i gradijenata
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Ažuriranje parametara
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config


    def _save_checkpoint(self):
        if self.checkpoint_name is None: return
        checkpoint = {
          'model': self.model,
          'update_rule': self.update_rule,
          'lr_decay': self.lr_decay,
          'optim_config': self.optim_config,
          'batch_size': self.batch_size,
          'num_train_samples': self.num_train_samples,
          'num_val_samples': self.num_val_samples,
          'epoch': self.epoch,
          'loss_history': self.loss_history,
          'train_acc_history': self.train_acc_history,
          'val_acc_history': self.val_acc_history,
        }
        filename = '%s_epoch_%d.pkl' % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)


    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Provjerite tačnost modela na priloženim podacima.

         Ulazi:
         - X: Niz podataka, oblika (N, d_1, ..., d_k)
         - y: Niz labela, oblika (N,)
         - num_samples: ako nije None, samo se model testira na num_samples broju podataka.
         - batch_size: Podijelite X i y na serije ove veličine kako biste izbjegli upotrebu
           previše memorije.

         Rezultat:
         - acc: Skalar koji predstavlja odnos tačno klasifikovanih podataka
           i ukupnog boja podataka.
        """

        # Uzimanje num_samples podataka
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Računanje predikcija po batch-u
        num_batches = N // batch_size
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
        Funkcija za optimizaciju modela.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Štampanje funkcije gubitka tokom treninga
            if self.verbose and t % self.print_every == 0:
                print('(Iteracija %d / %d) funkcija gubitka: %f' % (
                       t + 1, num_iterations, self.loss_history[-1]))

            # Na kraju svake epohe treba povećati brojač i smanjiti learning rate
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # Provjertiti tačnost na trening i validacionom skupu podataka za prvu iteraciju,
            # za poslednju iteraciju i posle svake epohe.
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train,
                    num_samples=self.num_train_samples)
                val_acc = self.check_accuracy(self.X_val, self.y_val,
                    num_samples=self.num_val_samples)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print('(Epoha %d / %d) trening tačnost: %f; validaciona tačnost: %f' % (
                           self.epoch, self.num_epochs, train_acc, val_acc))

                # Čuvati najbolji model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # Na kraju treninga postaviti najbolje parametre kao finalne parametre modela
        self.model.params = self.best_params
