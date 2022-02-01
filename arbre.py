import itertools
from numbers import Number
from random import random
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.core.dtypes.common import is_string_dtype


class TreeNode:
    # Initialisation de la classe, avec quelques paramètres par défaut
    def __init__(self, min_samples_split=2, max_depth=None, seed=2,
                 verbose=False):
        self.children = {}
        self.decision = None
        self.split_feat_name = None  # Fonction de fractionnement
        self.threshold = None
        # Nombre minimum d'échantillons pour faire un fractionnement
        self.min_samples_split = min_samples_split

        self.max_depth = max_depth
        self.seed = seed
        self.verbose = verbose


# Déclaration de fonctions
# ########################################################
# Cette fonction valide le type de données et la longueur égale des données et de la cible

def validate_data(self, data, target):
    # Validation du type de données pour l'échantillon (X)
    if not isinstance(data, (list, pd.core.series.Series, np.ndarray, pd.DataFrame)):
        return False
    # Validation du type de données pour l'échantillon (y)
    if not isinstance(target, (list, pd.core.series.Series, np.ndarray)):
        return False
    if len(data) != len(target):
        return False
    return True


    # C'est une fonction récursive qui sélectionne la décision si possible.
    # Sinon, créez des nœuds enfants.


def recursive_generate_tree(self, sample_data, sample_target, current_depth):
    # S'il n'y a qu'un seul résultat possible, sélectionnez-le comme décision
    if len(sample_target.unique()) == 1:
        self.decision = sample_target.unique()[0]
        # Si l'échantillon a moins de min_samples_split, sélectionnez la classe majoritaire
    elif len(sample_target) < self.min_samples_split:
        self.decision = self.get_maj_class(sample_target)

        # Si la profondeur de la branche courante est égale à max_depth \
        # sélectionnez la classe majoritaire
    elif current_depth == self.max_depth:
        self.decision = self.get_maj_class(sample_target)

        best_attribute, best_threshold, splitter = self.split_attribute(sample_data, sample_target)
        self.children = {}
        self.split_feat_name = best_attribute
        self.threshold = best_threshold
        current_depth += 1

        for v in splitter.unique():
            index = splitter == v  # Sélectionnez les index de chaque classe
            # S'il y a des données dans le nœud, créez un nouveau nœud d'arborescence avec cette partition
            if len(sample_data[index]) > 0:
                self.children[v] = TreeNode(min_samples_split=self.min_samples_split,
                                            max_depth=self.max_depth,
                                            seed=self.seed,
                                            verbose=self.verbose)
                self.children[v].recursive_generate_tree(sample_data[index], sample_target[index],
                                                         current_depth)

            else:
                self.children[v] = TreeNode(min_samples_split=self.min_samples_split,
                                            max_depth=1,
                                            seed=self.seed,
                                            verbose=self.verbose)
                self.children[v].recursive_generate_tree(sample_data, sample_target, current_depth=1)

                # Cette fonction définit quel est le meilleur attribut à diviser (chaîne \
                # ou continu)


def split_attribute(self, sample_data, sample_target):
    global threshold
    info_gain_max = -1 * float("inf")

    # Création d'une série vide pour stocker la variable dans laquelle la division est basée
    splitter = pd.Series(dtype='str')
    best_attribute = None
    best_threshold = None
    for attribute in sample_data.keys():
        if is_string_dtype(sample_data[attribute]):
            # Calculer le gain d'informations en utilisant cet attribut pour diviser la cible
            aig = self.compute_info_gain(sample_data[attribute], sample_target)
            if aig > info_gain_max:
                splitter = sample_data[attribute]
                info_gain_max = aig
                best_attribute = attribute
                best_threshold = None
                # Si l'attribut est un continu
            else:
                # Trier la variable continue dans un ordre croissant. Modifier l'ordre cible \
                # basé sur cela
                sorted_index = sample_data[attribute].sort_values(ascending=True).index
                sorted_sample_data = sample_data[attribute][sorted_index]
                sorted_sample_target = sample_target[sorted_index]
                # Itérer entre chaque échantillon, sauf le dernier
                for j in range(0, len(sorted_sample_data) - 1):
                    classification = pd.Series(dtype='str')

                    # Si deux échantillons consécutifs ne sont pas identiques,
                    # utilisez sa moyenne comme Un seuil
                    if sorted_sample_data.iloc[j] != sorted_sample_data.iloc[j + 1]:
                        threshold = (sorted_sample_data.iloc[j] +
                                     sorted_sample_data.iloc[j + 1]) / 2
                        classification = sample_data[attribute] > threshold
                        classification[classification] = 'greater'
                        classification[classification == False] = 'less'

                        # Calculer le gain d'information en utilisant la variable précédente \
                        # (maintenant catégorique)
                        aig = self.compute_info_gain(classification, sample_target)

                        if aig >= info_gain_max:
                            splitter = classification
                            info_gain_max = aig
                            best_attribute = attribute
                            best_threshold = threshold
            # Si verbeux est vrai, affiche le résultat de la division
            if self.verbose:
                if is_string_dtype(sample_data[best_attribute]):
                    print(f"Split by {best_attribute}, IG: {info_gain_max:.2f}")
                else:
                    print(
                        f"Split by {best_attribute}, at {threshold}, IG: {info_gain_max:.2f}")
            return best_attribute, best_threshold, splitter

    # Cette fonction calcule l'entropie basée sur la distribution de \
    # le fractionnement cible


def compute_entropy(self, sample_target_split):
    # S'il n'y a qu'une seule classe, l'entropie est 0
    if len(sample_target_split) < 2:
        return 0

    # Sinon calculer l'entropie
    else:
        freq = np.array(sample_target_split.value_counts(normalize=True))

        return -(freq * np.log2(freq + 1e-6)).sum()


# Cette fonction calcule le gain d'information en utilisant un \
# attribut pour diviser la cible
def compute_info_gain(self, sample_attribute, sample_target):
    values = sample_attribute.value_counts(normalize=True)
    split_ent = 0

    for v, fr in values.iteritems():
        # Calculer l'entropie pour l'échantillon cible correspondant à la classe
        index = sample_attribute == v
        sub_ent = self.compute_entropy(sample_target[index])

        split_ent += fr * sub_ent

    ent = self.compute_entropy(sample_target)
    # Renvoie le gain d'information du grand écart
    return ent - split_ent


# Cette fonction sélectionne la classe majoritaire de la cible pour prendre une décision
def get_maj_class(self, sample_target):
    freq = sample_target.value_counts().sort_values(ascending=False)

    # Sélectionnez le nom de la classe (classes) qui a le nombre maximum d'enregistrements
    maj_class = freq.keys()[freq == freq.max()]

    if len(maj_class) > 1:
        decision = maj_class[random.Random(self.seed).randint(0, len(maj_class) - 1)]

    else:
        decision = maj_class[0]
    return decision


# cette fonction retourne la classe ou prédiction donnée un x
def predict(self, sample):
    if self.decision is not None:
        if self.verbose:
            print("Decision:", self.decision)
        return self.decision
    else:
        # Sélectionnez la valeur de l'attribut grand écart dans les données
        attr_val = sample[self.split_feat_name]
        if self.verbose:
            print('attr_val')
        # Si la valeur de la fonctionnalité n'est pas numérique, accédez simplement à \
        # nœud enfant correspondant et impression
        if not isinstance(attr_val, Number):
            child = self.children[attr_val]
            if self.verbose:
                print("Testing ", self.split_feat_name, "->", attr_val)
        # Si la valeur est numérique, voyez si elle est supérieure ou inférieure à \
        # seuil
        else:
            if attr_val > self.threshold:
                child = self.children['greater']
                if self.verbose:
                    print("Testing ", self.split_feat_name, "->",
                          'greater than ', self.threshold)
            else:
                child = self.children['less']
                if self.verbose:
                    print("Testing ", self.split_feat_name, "->",
                          'less than or equal', self.threshold)
        return child.predict(sample)


# Cette fonction effectue un réglage hyperparathyroidism en utilisant la validation croisée

def cross_validation(x, y, model, n_folds, params, seed=4):
    # Création de combinaisons de tous les hyperparathyroidism
    keys, values = zip(*params.items())
    params = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # Initialisez une liste pour stocker la précision de chaque ensemble de paramètres
    accuracy = []
    for j in range(0, len(params)):
        indexes = np.array(x.index)
        random.Random(seed).shuffle(indexes)
        folds = np.array_split(indexes, n_folds)
        acc = []  # Initialiser une précision locale pour chaque pli
        # Calculer la précision de chaque pli en utilisant le premier comme ensemble de test
        for k in range(1, n_folds):
            model.set_params(params[j])
            # Entraînez le modèle avec le pli
            model.fit(x.loc[folds[k]], y.loc[folds[k]])
            prediction = x.loc[folds[1]].apply(lambda row: model.predict(row), axis=1)
            acc.append(sum((prediction == y.loc[folds[1]])) / len(y.loc[folds[1]]))
        # Calculer la précision de tous les plis pour l'ensemble de paramètres
        accuracy.append(sum(acc) / len(acc))
        print(f"Parameters: {params[j]}, Accuracy: {accuracy[j]}")
        seed = seed + 1

    # Sélectionner l'ensemble des meilleurs hyperparathyroidism
    max_value = max(accuracy)
    max_index = accuracy.index(max_value)
    print('Best hyper parameters:')


# Function Main
# ########################################################

def main(filename: str = 'dataset.csv'):
    # Chargement des données
    df = pd.read_csv(filename)
    print("L'entête du dataframe: \n\n", df.head())


"""
    # Convertir les données 
    data = convert_data(df)
    data = np.array(data)

 # Mélanger les données
    np.random.shuffle(data)

 # Extraction des données
    x = data[:, 0:13]
    y = data[:, 13]

    # Normalisation des données
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x = (x - mean) / std

    # division des données
    div_index = int(0.80 * len(x))
    x_train = x[0:div_index]
    y_train = y[0:div_index]
    x_test = x[div_index:]
    y_test = y[div_index:]




   main()
"""
"""
import pandas as pd

# Load data.
df = pd.read_csv("wifi_localization.txt", delimiter="\t")
data = df.to_numpy()
X = data[:, :-1] # all columns but the last
y = data[:, -1] - 1 # expected to be from 0 to n_classes - 1

# Fit data.
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X, y)

# Visualize.
clf.debug(
    features_names=["Wifi {}".format(i) for i in range(1, 8)],
    target_names=["Room {}".format(i) for i in range(1, 5)],
)
"""
