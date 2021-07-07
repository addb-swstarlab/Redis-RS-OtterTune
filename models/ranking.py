
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import lasso_path
from xgboost import XGBRegressor

from .base import ModelBase

class Ranking(ModelBase):
    def __init__(self,mode):
        self.mode = mode
        if mode == "lasso":
            self.init_lasso()
        elif mode == "RF":
            self.init_RF()
        elif mode == 'XGB':
            self.init_XGB()

    def _reset(self):
        """Resets all attributes (erases the model)"""
        self.feature_labels_ = None
        self.alphas_ = None
        self.coefs_ = None
        self.rankings_ = None

    def init_lasso(self):
        """Lasso:

        Computes the Lasso path using Sklearn's lasso_path method.


        See also
        --------
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html


        Attributes
        ----------
        feature_labels_ : array, [n_features]
                        Labels for each of the features in X.

        alphas_ : array, [n_alphas]
                The alphas along the path where models are computed. (These are
                the decreasing values of the penalty along the path).

        coefs_ : array, [n_outputs, n_features, n_alphas]
                Coefficients along the path.

        rankings_ : array, [n_features]
                The average ranking of each feature across all target values.
        """
        self.feature_labels_ = None
        self.alphas_ = None
        self.coefs_ = None
        self.rankings_ = None

    def init_RF(self):

        """
        RandomForest:

        Computes the RandomForest using Sklearn's RandomForestRegressor method.


        See also
        --------
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html


        Attributes
        ----------
        feature_labels_ : array, [n_features]
                        Labels for each of the features in X.

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

    def init_XGB(self, n_estimators = 100, max_depth = 7, learning_rate = 0.08, booster='gbtree', base_score = 0.5, colsample_bytree = 1):
        """
        XGBOOST:

        Computes the xgboost using XGBRegressor.

        See also
        --------
        https://xgboost.readthedocs.io/en/latest/python/python_api.html


        Attributes
        ----------
        feature_labels_ : array, [n_features]
                        Labels for each of the features in X.

        rankings_ : array, [n_features]
                The average ranking of each feature across all target values.
                
        Init xgboost model using XGBRegressor

        Parameters
        ----------
        n_estimators : int, default = 100
                            Number of gradient boosted trees. Equivalent to number of boosting rounds.
        max_depth : int, optional
                            Maximum tree depth for base learners.
        learning_rate  : float, optional
                            Boosting learning rate.
        booster  : str, optional
                            Specify which booster to use: gbtree, gblinear or dart.                            
        base_score  : float, optional
                            The initial prediction score of all instances, global bias.                                                   
        colsample_bytree  : float, optional
                            Subsample ratio of columns when constructing each tree.

        ##TOBE:
        # can add more parameters                            
        ----------
        """
        self.feature_labels_ = None
        self.rankings_ = None
        self.model = XGBRegressor(n_estimators = n_estimators,
                                 max_depth = max_depth,
                                 learning_rate = learning_rate,
                                 booster = booster,
                                 base_score = base_score,
                                 colsample_bytree = colsample_bytree,
                                 )

    def lasso_fit(self, X, y, feature_labels, estimator_params=None):
        """Computes the Lasso path using Sklearn's lasso_path method.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data (the independent variables).

        y : array-like, shape (n_samples, n_outputs)
            Training data (the output/target values).

        feature_labels : array-like, shape (n_features)
                         Labels for each of the features in X.

        estimator_params : dict, optional
                           The parameters to pass to Sklearn's Lasso estimator.


        Returns
        -------
        self
        """
        self._reset()
        if estimator_params is None:
            estimator_params = {}
        self.feature_labels_ = feature_labels

        alphas, coefs, _ = lasso_path(X, y, **estimator_params)
        self.alphas_ = alphas.copy()
        self.coefs_ = coefs.copy()

        # Rank the features in X by order of importance. This ranking is based
        # on how early a given features enter the regression (the earlier a
        # feature enters the regression, the MORE important it is).
        feature_rankings = [[] for _ in range(X.shape[1])]
        for target_coef_paths in self.coefs_:
            for i, feature_path in enumerate(target_coef_paths):
                entrance_step = 1
                for val_at_step in feature_path:
                    if val_at_step == 0:
                        entrance_step += 1
                    else:
                        break
                feature_rankings[i].append(entrance_step)
        self.rankings_ = np.array([np.mean(ranks) for ranks in feature_rankings])
        return self

    def rf_fit(self,X,y,feature_labels):
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

    def xgb_fit(self,X,y,feature_labels):
        """Computes the xgboost using XGBRegressor.

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
        feature_importance = self.model.feature_importances_
        self.rankings_ = []
        assert len(self.feature_labels_) == len(feature_importance)
        for label, imp in zip(self.feature_labels_,feature_importance):
            self.rankings_.append((float(imp),label))
        return self

    def lasso_get_ranked_features(self):
        if self.rankings_ is None:
            raise Exception("No lasso path has been fit yet!")

        rank_idxs = np.argsort(self.rankings_)
        return [self.feature_labels_[i] for i in rank_idxs]

    def rf_get_ranked_features(self):
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

    def xgb_get_ranked_feature(self):
        """
        Compute feature_importance using trained XGBR model by rankings_.
        Sort by feature importance descending.

        Returns
        -------
        sorted features(knobs)
        """
        if self.rankings_ is None:
            raise Exception("No XGBR has been fit yet")
        
        rank_idxs = sorted(self.rankings_,key= lambda x : x[0],reverse=True)
        return [v for _,v in rank_idxs]

    def rf_get_ranked_importance(self):
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

    def xgb_get_ranked_importance(self):
        """
        Compute feature_importance using trained XGBR model by rankings_.
        Sort by feature importance descending.

        Returns
        -------
        sorted (features(knobs),importance)
        """
        if self.rankings_ is None:
            raise Exception("No XGBR has been fit yet")
        
        rank_idxs = sorted(self.rankings_,key= lambda x : x[0],reverse=True)
        return [(v,i) for i,v in rank_idxs]

    def fit(self,X,y,feature_labels):
        if self.mode == "lasso":
            return self.lasso_fit(X,y,feature_labels)
        elif self.mode == "RF":
            return self.rf_fit(X,y,feature_labels)
        elif self.mode == "XGB":
            return self.xgb_fit(X,y,feature_labels)
    
    def get_ranked_features(self):
        if self.mode == "lasso":
            return self.lasso_get_ranked_features()
        elif self.mode == "RF":
            return self.rf_get_ranked_features()
        elif self.mode == "XGB":
            return self.xgb_get_ranked_feature()
    def get_ranked_importance(self):
        if self.mode == "lasso":
            return None
        elif self.mode == "RF":
            return self.rf_get_ranked_importance()
        elif self.mode == "XGB":
            return self.xgb_get_ranked_importance()
    
