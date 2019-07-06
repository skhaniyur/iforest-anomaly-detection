
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def c_path(psi):
    if psi > 2:
        c = 2 * (np.log(psi - 1) + 0.5772156649) - (2 * (psi - 1) / psi)
    elif psi == 2:
        c = 1
    else:
        c = 0
    return c


def PathLength(x, T, e):
    if isinstance(T, LeafNode):
        return e + c_path(T.size)

    a = T.q
    if x[a] < T.p:
        return PathLength(x, T.left, e + 1)
    else:
        return PathLength(x, T.right, e + 1)


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.n_trees = n_trees
        self.psi = sample_size
        self.trees = []

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        for i in range(self.n_trees):
            # X_sample = X[np.random.randint(low=0, high=len(X), size=self.psi, replace=False)]
            X_sample = X[np.random.choice(X.shape[0], size=self.psi, replace=False), :]
            height_limit = np.log2(self.psi)

            tree = IsolationTree(height_limit=height_limit)
            tree.fit(X=X_sample, improved=improved)
            self.trees.append(tree)

        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        lengths = np.zeros([len(X), self.n_trees])
        x_counter = 0

        for x in X:
            e = 0
            t_counter = 0
            for tree in self.trees:
                lengths[x_counter, t_counter] = PathLength(x, tree.root, e)
                t_counter += 1
            x_counter += 1

        return np.mean(lengths, axis=1)

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        path_lengths = self.path_length(X)
        s = np.divide(path_lengths, c_path(self.psi))
        scores = 2 ** (s * -1)
        return scores

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        y_pred = (scores > threshold).astype(int)
        return y_pred

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        return self.predict_from_anomaly_scores(self.anomaly_score(X), threshold)


class LeafNode():
    def __init__(self, size, depth):
        self.size = size
        self.depth = depth


class DeciNode():
    def __init__(self, q, p, left, right):
        self.q = q
        self.p = p
        self.left = left
        self.right = right


class IsolationTree:
    def __init__(self, height_limit):
        self.root = None
        self.n_nodes = 0
        self.h_lim = height_limit

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        self.root = self.fit_(X, level=0, improved=improved)
        return self.root

    def fit_(self, X, level, improved):
        if improved:
            z = 3
        else:
            z = 1

        if level >= self.h_lim or len(X) <= z or all(np.all(X == X[0, :], axis=1)):
            self.n_nodes += 1
            return LeafNode(size=len(X), depth=level)
        else:
            Q = X.shape[1]
            q = np.random.randint(low=0, high=Q)
            p = np.random.uniform(low=min(X[:, q]), high=max(X[:, q]), size=1)[0]

            left_index = np.where(X[:, q] < p, True, False)
            X_left = X[left_index]
            X_right = X[~left_index]

            if len(X_left) == 0 or len(X_right) == 0:
                self.n_nodes += 1
                return LeafNode(size=len(X), depth=level)

            self.n_nodes += 1
            return DeciNode(q=q, p=p, left=self.fit_(X_left, level+1, improved=improved),
                            right=self.fit_(X_right, level+1, improved=improved))


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold = 1.0

    while threshold > 0:
        y_pred = (scores > threshold).astype(int)
        cm = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = cm.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        if TPR >= desired_TPR:
            return threshold, FPR

        threshold -= 0.01

    return None

