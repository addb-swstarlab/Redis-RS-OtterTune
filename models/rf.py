from sklearn.ensemble import RandomForestRegressor

class RFR(object):
    """RandomForest:

    Computes the RandomForest using Sklearn's RandomForestRegressor method.


    See also
    --------
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html


    Attributes
    ----------
    feature_labels_ : array, [n_features]
                      Labels for each of the features in X.
    """
    def __init__(self):
        """
        Init RandomForestRegressor model using sklearn.ensemble.RadnomForestRegressor

        Parameters
        ----------
        n_estimators : int, default = 100
                           The number of trees in the forest.

        ##TOBE:
        # add more parameters                            
        ----------
        """
        self.feature_labels_ = None
        self.model = RandomForestRegressor(n_estimators=100)

    def _reset(self):
        """Resets all attributes (erases the model)"""
        self.feature_labels_ = None

    def fit(self,X,y,feature_labels):
        """Computes the RandomForest using Sklearn's RandomForestRegressor method.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data (the independent variables).

        y : array-like, shape (n_samples, n_outputs)
            Training data (the output/target values).

        feature_labels : array-like, shape (n_features)
                         Labels for each of the features in X.
        Returns
        -------
        self
        """
        self._reset()

        self.feature_labels_ = feature_labels

        self.model.fit(X,y)
        return self

    def get_ranked_features(self):
        """
        Compute feature_importance using trained RF model.
        Sort by feature importance descending.

        Returns
        -------
        sorted features(knobs)
        """
        sorted_idx = self.model.feature_importances_.argsort()
        features = []
        for idx in sorted_idx:
            features.append((self.feature_labels_[idx],self.model.feature_importances_[idx]))
        features = sorted(features,key=lambda x: x[1],reverse=True)
        return [i for i,_ in features]

    def get_ranked_importance(self):
        """
        Compute feature_importance using trained RF model.
        Sort by feature importance descending.

        Returns
        -------
        sorted (features(knobs),importance)
        """
        sorted_idx = self.model.feature_importances_.argsort()
        features = []
        for idx in sorted_idx:
            features.append((self.feature_labels_[idx],self.model.feature_importances_[idx]))
        features = sorted(features,key=lambda x: x[1],reverse=True)
        return [(i,v) for i,v in features]

