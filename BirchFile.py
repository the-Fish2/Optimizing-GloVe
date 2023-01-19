# Authors: Manoj Kumar <manojkumarsivaraj334@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Joel Nothman <joel.nothman@gmail.com>
# License: BSD 3 clause

import warnings
import numpy as np
from numbers import Integral, Real
from scipy import sparse
from math import sqrt

from sklearn.metrics.pairwise import cosine_distances

from sklearn.metrics import pairwise_distances_argmin

from sklearn.base import (
    TransformerMixin,
    ClusterMixin,
    BaseEstimator,
    _ClassNamePrefixFeaturesOutMixin,
)


from sklearn.utils.extmath import row_norms

from param_validation import Interval

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import AgglomerativeClustering
from sklearn._config import config_context

from scipy.spatial.distance import cosine

