from lightgbm import LGBMRegressor
class LGBMR(object):
    """LGB<:
    Computes the lgbm using LGBMRegressor.
    See also
    --------
    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
    Attributes
    ----------
    feature_labels_ : array, [n_features]
                      Labels for each of the features in X.
    rankings_ : array, [n_features]
             The average ranking of each feature across all target values.
    """
    def __init__(self,n_estimators = 100, max_depth = 7, learning_rate = 0.08, booster='gbdt', colsample_bytree = 1):
        """
        Init lgbm model using LGBMRegressor
        Parameters
        ----------
        n_estimators : int, default = 100
                            Number of gradient boosted trees. Equivalent to number of boosting rounds.
        max_depth : int, optional
                            Maximum tree depth for base learners.
        learning_rate  : float, optional
                            Boosting learning rate.
        boosting_type : str, optional
                            Specify which booster to use: gbtree, gblinear or dart.                                                                          
        colsample_bytree  : float, optional
                            Subsample ratio of columns when constructing each tree.
        ##TOBE:
        # can add more parameters                            
        ----------
        """
        self.feature_labels_ = None
        self.rankings_ = None
        self.model1 = LGBMRegressor(
                                 n_estimators = n_estimators,
                                 max_depth = max_depth,
                                 learning_rate = learning_rate,
                                 boosting_type = booster,
                                 colsample_bytree = colsample_bytree,
                                 )
        self.model2 = LGBMRegressor(
                                 n_estimators = n_estimators,
                                 max_depth = max_depth,
                                 learning_rate = learning_rate,
                                 boosting_type = booster,
                                 colsample_bytree = colsample_bytree,
                                 )
        self.model3 = LGBMRegressor(
                                 n_estimators = n_estimators,
                                 max_depth = max_depth,
                                 learning_rate = learning_rate,
                                 boosting_type = booster,
                                 colsample_bytree = colsample_bytree,
                                 )

    def _reset(self):
        """Resets all attributes (erases the model)"""
        self.feature_labels_ = None
        self.rankings_ = None

    def fit(self,X,y,feature_labels):
        """Computes the xgboost using LGBMRegressor.
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
        print(y)
        self.model1.fit(X,y[:,0])
        self.model2.fit(X,y[:,1])
        self.model3.fit(X,y[:,2])
        feature_importance = self.model1.feature_importances_+self.model2.feature_importances_+self.model3.feature_importances_
        self.rankings_ = []
        assert len(self.feature_labels_) == len(feature_importance)
        for label, imp in zip(self.feature_labels_,feature_importance):
            self.rankings_.append((float(imp),label))
        return self

    def get_ranked_knobs(self):
        """
        Compute feature_importance using trained LGBMR model by rankings_.
        Sort by feature importance descending.
        Returns
        -------
        sorted features(knobs)
        """
        if self.rankings_ is None:
            raise Exception("No LGBMR has been fit yet")
        
        rank_idxs = sorted(self.rankings_,key= lambda x : x[0],reverse=True)
        return [v for _,v in rank_idxs]

    def get_ranked_importance(self):
        """
        Compute feature_importance using trained LGBMR model by rankings_.
        Sort by feature importance descending.
        Returns
        -------
        sorted (features(knobs),importance)
        """
        if self.rankings_ is None:
            raise Exception("No LGBMR has been fit yet")
        
        rank_idxs = sorted(self.rankings_,key= lambda x : x[0],reverse=True)
        return [(v,i) for i,v in rank_idxs]
