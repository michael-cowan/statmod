import numpy as np
import re


class Leaf(object):
    """Leaf node that terminates Decision Tree branches

    Arguments:
    parent (Node): parent Decision Tree Node
    y (np.ndarray): target variables selected by parent Node
    node_type (str): type of node: root or (left|right)-(decision|leaf)
    """
    def __init__(self, parent, y, node_type):
        self.parent = parent
        self.y = y

        self.node_type = node_type + '-leaf'

        # compute the mode = predicted output of Leaf node
        self.mode = np.argmax(np.bincount(y))

        # create prediction str using parent's target names
        self.prediction = self.parent.target_names[self.mode]

        self.target_names = target_names
        self.depth = parent.depth + 1

    def __str__(self):
        val = f'|{self.prediction} ({len(self.y)})|'

        # ensure str representation does not exceed 25 characters
        if len(val) > 25:
            # truncate prediction name if over 25 chars
            val = val.replace(self.prediction,
                              self.prediction[:-(len(val) - 25)])

        return val.center(25)

    def predict(self, x):
        """Return the mode of the target variables
        - mode = prediction of Decision Tree
        """
        return self.mode


class Node(object):
    """Decision node that employs ID3 algorithm to select decision criterion
    - Recursively creates children Nodes

    Arguments:
    parent (Node): parent Decision Tree Node
    X (np.ndarray): training data
    y (np.ndarray): target variables

    Keyword Arguments:
    max_depth (int): max depth of Decision nodes past the root node.
                     (Default: 3)
    categorical (list): list of data column indices that have categorical data
                        - needed to differentiate decision type (< vs ==)
                        (Default: [])
    node_type (str): type of node: root or (left|right)-(decision|leaf)
                     (Default: None)
    """
    num_nodes = 0
    total_depth = 0

    def __init__(self, parent, X, y, max_depth=3, categorical=[],
                 node_type=None):
        # parent node
        self.parent = parent

        # if no parent, node = root
        if self.parent is None:
            self.node_type = 'root'
            # track depth
            self.depth = 0

            # maximum depth allowed
            self.max_depth = max_depth

            # list of column indices that are categorical
            # if index not in list, assume continuous
            self.categorical = categorical
        else:
            # increment depth based on parent
            self.depth = self.parent.depth + 1
            Node.total_depth = max(self.depth, Node.total_depth)

            # also get other KWargs from parent
            self.max_depth = self.parent.max_depth
            self.categorical = self.parent.categorical
            self.feature_names = self.parent.feature_names
            self.target_names = self.parent.target_names
            # node type = left or right - decision node
            self.node_type = node_type + '-decision'

        # input and target data
        self.X = X
        self.y = y

        # calculate entropy of parent data
        # (before using this node to divide the data)
        parent_prob = np.bincount(y) / len(y)
        parent_log = np.log(parent_prob, out=np.zeros_like(parent_prob),
                            where=parent_prob != 0)
        self.parent_entropy = -parent_prob @ parent_log

        # get the optimal decision variable / cutoff
        # based on minimizing Entropy
        self.entropy = None
        self.cutoff = None
        self.get_decision_var()

        # track number of nodes
        Node.num_nodes += 1

        # make Node / Leaf children
        self.make_children()

    def __str__(self):
        name = self.feature_names[self.decision_var]
        val = f'[{name} < {self.cutoff:.3g}]'

        # ensure str representation does not exceed 25 characters
        if len(val) > 25:
            # truncate feature name if over 25 chars
            val = val.replace(name, name[:-(len(val) - 25)])

        return val.center(25)

    def get_decision_var(self):
        var_info_gain = []
        for var in range(self.X.shape[1]):
            # sort the data if decision variable is not categorical
            if var not in self.categorical:
                ids = np.argsort(self.X[:, var])
                self.X = self.X[ids, :]
                self.y = self.y[ids]
            var_info_gain.append(self.get_max_info_gain(var))
        max_gain = max(var_info_gain)
        self.decision_var = var_info_gain.index(max_gain)
        self.info_gain, self.cutoff = max_gain
        self.split_data(self.decision_var, self.cutoff)

    def get_max_info_gain(self, var):
        info_gains = [[0, 0]]
        for cut_i in range(1, len(self.y)):
            if self.X[cut_i, var] != self.X[cut_i - 1, var]:
                ig = self.parent_entropy - self._entropy(var, cut_i)
                info_gains.append([ig, cut_i])

        info_gain, cut_i = max(info_gains)
        cutoff = self.X[cut_i, var]
        return [info_gain, cutoff]

    def _entropy(self, var, cut_i):
        self.split_data(var, cut_i, is_sorted=True)

        prob_leftright = np.array([len(self.y_left),
                                   len(self.y_right)]) / len(self.y)

        prob_lefty = np.bincount(self.y_left) / len(self.y_left)
        prob_righty = np.bincount(self.y_right) / len(self.y_right)

        logleft = np.log(prob_lefty, out=np.zeros_like(prob_lefty),
                         where=prob_lefty != 0)
        logright = np.log(prob_righty, out=np.zeros_like(prob_righty),
                          where=prob_righty != 0)

        H = prob_leftright @ [(-prob_lefty * logleft).sum(),
                              (-prob_righty * logright).sum()]

        return H

    def make_children(self):
        for side in ['left', 'right']:
            x = getattr(self, f'x_{side}')
            y = getattr(self, f'y_{side}')

            # calculate entropy of data
            prob = np.bincount(y) / len(y)
            log = np.log(prob, out=np.zeros_like(prob),
                         where=prob != 0)
            e = -prob @ log

            # decide whether to create another decision node or
            # a leaf node to terminate the branch
            if self.depth + 1 == self.max_depth or len(y) < 2 or e < 0.2:
                setattr(self, side, Leaf(self, y, node_type=side))
            else:
                setattr(self, side, Node(self, x, y, node_type=side))

    def predict(self, x):
        if x[self.decision_var] < self.cutoff:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def split_data(self, var, criterion, is_sorted=False):
        if var in self.categorical:
            div = self.X[:, var] == criterion
            self.x_left = self.X[div]
            self.y_left = self.y[div]

            self.x_right = self.X[~div]
            self.y_right = self.y[~div]

        elif is_sorted:
            self.x_left, self.x_right = np.vsplit(self.X, [criterion])
            self.y_left, self.y_right = np.split(self.y, [criterion])
        else:
            div = self.X[:, var] < criterion
            self.x_left = self.X[div]
            self.y_left = self.y[div]

            self.x_right = self.X[~div]
            self.y_right = self.y[~div]


class DecisionTree(Node):
    """NumPy-based Decision Tree Classifier. Uses ID3 algorithm for training
    - ID3 (Iterative Dichotomiser 3): greedy algorithm that selects decision
      criteria using Entropy and Information Gain
    - Inherits the Node class: Tree is iteratively built through make_children

    Arguments:
    X (np.ndarray): training data
    y (np.ndarray): target variables

    Keyword Arguments:
    max_depth (int): max depth of Decision nodes past the root node.
                     (Default: 3)
    categorical (list): list of data column indices that have categorical data
                        - needed to differentiate decision type (< vs ==)
                        (Default: [])
    feature_names (list): list of column names for data
                          (Default: None = 'X[<int>]')
    target_names (list): list of column names for target
                         (Default: None = 'y[<int>]')
    """
    def __init__(self, X, y, max_depth=3, categorical=[], feature_names=None,
                 target_names=None):
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.categorical = categorical

        self.feature_names = feature_names
        if feature_names is None:
            self.feature_names = [f'X[{i}]' for i in range(X.shape[0])]

        self.target_names = target_names
        if target_names is None:
            self.target_names = [f'y[{i}]' for i in range(X.shape[0])]

        # reset Node.total_depth and Node.num_nodes to 0
        # (in case multiple Trees are initialized)
        Node.total_depth = 0
        Node.num_nodes = 0

        # call Node's init to train the tree
        super().__init__(None, X, y, max_depth, categorical)

        # get total_depth and num_nodes from Node class attribute
        self.total_depth = Node.total_depth + 1
        self.num_nodes = Node.num_nodes

        # calculate str for each depth in tree
        self._levels = self._make_tree_str(self)

    def __len__(self):
        """Returns total number of decision nodes in tree"""
        return self.num_nodes

    def __str__(self):
        """Print the tree one level (depth) at a time"""
        return '\n'.join(self._levels)

    def _make_tree_str(self, node=None, i=75, _levels=None):
        """Recursive approach to generate string of Decision Tree
        - string for each depth level is updated in <levels> list

        Keyword Arguments:
        node: current node to add to levels list
              (Default: None - starts with root node (self))
        i (int): value used to center node's str
                 (Default: 50)
        _levels (list): list of strings for each depth level in tree
                        (Default: list of empty strings)
        """
        if node is None:
            node = self
        if _levels is None:
            _levels = [''] * (self.total_depth + 1)

        # if root node, use super()'s __str__ method
        if node.depth == 0:
            _levels[0] = super().__str__()
            _levels[0] = _levels[0].rjust(i)
        else:
            # if nodes already present in depth string, use - separator
            if _levels[node.depth]:
                use = max(0, i - len(_levels[node.depth]))
            else:
                use = i

            _levels[node.depth] += f'{str(node)}'.rjust(use)

            # add '-' characters between left-right nodes
            if 'right' in node.node_type:
                # use regex to find space between left and right nodes
                for match in re.findall('[|\\]] +[|\\[]',
                                        _levels[node.depth]):
                    n = match.replace(' ', '-')
                    _levels[node.depth] = _levels[node.depth
                                                  ].replace(match, n)

        if isinstance(node, Node):
            max_spread = self.max_depth * 5 + 5
            spread = max_spread - 10 * node.depth
            _levels = self._make_tree_str(node.left, i - spread,
                                          _levels=_levels)
            _levels = self._make_tree_str(node.right, i + spread,
                                          _levels=_levels)

        return _levels

    def predict(self, X):
        """Predict the category for an array of inputs

        Arguments:
        X (np.ndarray): array of input data

        Returns:
        (np.ndarray): predictions from trained Decision Tree
        """
        # cannot call super() in list comprehension?
        sup = super()
        yhat = np.array([sup.predict(x) for x in X])
        return yhat

    def score(self, X, y):
        """Computes the accuracy of the Decision Tree
        - Accuracy = (number correct) / (number of data)

        Arguments:
        X (np.ndarray): training data
        y (np.ndarray): target variables

        Keyword Arguments:
        (float): accuracy of Tree's predictions
        """
        yhat = self.predict(X)
        return (yhat == y).sum() / len(y)


if __name__ == '__main__':
    # use iris example dataset
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data['data'], data['target']
    feature_names = data['feature_names']
    target_names = data['target_names']

    # create and print decision tree
    tree = DecisionTree(X, y, feature_names=feature_names, max_depth=2,
                        target_names=target_names)
    print(tree)

    # print the # decision nodes and accuracy
    print(f'Num Target Classes: {len(np.unique(y))}')
    print(f'Num Decision Nodes: {len(tree)}')
    print(f'    Total Accuracy: {tree.score(X, y):.2%}')
