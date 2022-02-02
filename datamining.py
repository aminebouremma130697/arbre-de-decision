import math
import numpy as np
import pandas as pd

# Déclaration de constantes
CLASS_SETOSA = 1
CLASS_VIRGINICA = 2
CLASS_VERSICOLOR = 3


# Déclaration des classes
# ########################################################

class Model:
    def __init__(self):
        self.__x = []
        self.__y = []

    @staticmethod
    def distance(obs1, obs2) -> float:
        s = 0.0
        for i in range(0, len(obs1)):
            s = s + (obs1[i] - obs2[i]) ** 2
        return math.sqrt(s)

    def fit(self, x, y) -> None:
        self.__x = x
        self.__y = y

    def predict(self, obs) -> int:
        # calcul des distances entre obs et toutes les observations dans l'ensemble d'entrainement (x)
        distances = []
        for obs2 in self.__x:
            distances.append(Model.distance(obs, obs2))

        # indices des k plus petites valeurs
        idx = np.argpartition(distances, self.__k)

        # calcul de la classe la plus probable, parmi les k possibles
        classes_nb = {
            CLASS_SETOSA: 0,
            CLASS_VIRGINICA: 0,
            CLASS_VERSICOLOR: 0
        }
        k = self.__k
        for i in idx:
            if k <= 0:
                break
            k = k - 1
            classes_nb[self.__y[i]] += 1

        # retourner la clé correspondant à la classe la plus répandue dans classes_nb
        class_max = max(classes_nb, key=classes_nb.get)

        return class_max

    def predict_all(self, list_obs) -> list:
        r = []
        for obs in list_obs:
            r.append(self.predict(obs))
        return r


# Déclaration de fonctions
# ########################################################


def convert_species(species: str) -> int:
    """ Convertit une classe (chaîne de caractères) en un entier """
    if species == 'Iris-setosa':
        return CLASS_SETOSA
    elif species == 'Iris-versicolor':
        return CLASS_VERSICOLOR
    elif species == 'Iris-virginica':
        return CLASS_VIRGINICA
    else:
        raise ValueError('Mauvaise donnée: ' + species)


def convert_data(df) -> list:
    r = []
    for index, line in df.iterrows():
        obs = [
            float(line['SepalLengthCm']),
            float(line['SepalWidthCm']),
            float(line['PetalLengthCm']),
            float(line['PetalWidthCm']),
            convert_species(line['Species'])
        ]
        r.append(obs)

    return r


def accuracy(predictions, trues) -> float:
    total = 0
    for i in range(0, len(predictions)):
        if predictions[i] == trues[i]:
            total += 1
    return total / len(predictions)


# Fonction Main
# ########################################################

def main(filename: str = 'iris.csv'):
    # Chargement des données
    df = pd.read_csv(filename)
    print("L'entête du dataframe: \n\n", df.head())

    # Convertir les données
    data = convert_data(df)
    data = np.array(data)

    # Mélanger les données
    np.random.shuffle(data)

    # Extraction des données
    x = data[:, 0:4]
    y = data[:, 4]

    # Normalisation des données
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x = (x - mean) / std

    # division des données
    div_index = int(0.70 * len(x))
    x_train = x[0:div_index]
    y_train = y[0:div_index]
    x_test = x[div_index:]
    y_test = y[div_index:]

    # modèle 1
    mod1 = Model(1)
    mod1.fit(x_train, y_train)
    y_pred1 = mod1.predict_all(x_test)
    acc1 = accuracy(y_pred1, y_test)
    print('Accuracy 1 = ', acc1)

    # modèle 2
    mod2 = Model(3)
    mod2.fit(x_train, y_train)
    y_pred2 = mod2.predict_all(x_test)
    acc2 = accuracy(y_pred2, y_test)
    print('Accuracy 2 = ', acc2)

    #
    if acc1 > acc2:
        print('Le modèle 1 est meilleur')
    else:
        print('Le modèle 2 est meilleur')

    # validation croisée en 10 volets
    nb_volets = 10
    score_mod1 = 0  # nombre de fois où le mod1 est meilleur
    score_mod2 = 0  # nombre de fois où le mod2 est meilleur
    div_index = int(0.10 * len(x))
    for i in range(0, nb_volets):
        # calcul des indices à utiliser pour l'entrainement / validation
        validation_range = range(i * div_index, (i + 1) * div_index)
        train_range = list(set(range(len(x))) - set(validation_range))

        #
        x_train = x[train_range]
        y_train = y[train_range]
        x_val = x[validation_range]
        y_val = y[validation_range]

        #
        mod1.fit(x_train, y_train)
        acc1 = accuracy(mod1.predict_all(x_val), y_val)
        mod2.fit(x_train, y_train)
        acc2 = accuracy(mod2.predict_all(x_val), y_val)

        #
        if acc1 > acc2:
            score_mod1 += 1
        else:
            score_mod2 += 1

        if score_mod1 > score_mod2:
            print('Résultat de la validation croisée: Le modèle 1 est meilleur')
        else:
            print('Résultat de la validation croisée: Le modèle 2 est meilleur')


# Appelle de la fonction main
main()

# Réponse à la question 6:
# Non, nous pouvons uniquement statuer en ce qui concerne le jeu de données et la division utilisée.
# D'ailleurs, nous avons des résultats différents selon le mélange aléatoire réalisé.
# Pour avoir une réponse plus correcte, il faut utiliser une VALIDATION CROISÉE.
# drn pointdd
