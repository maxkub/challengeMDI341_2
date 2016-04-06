import xgboost as xgb
from hyperopt import fmin, tpe, hp, fmin, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import numpy as np

class XgbModel():
    
    def __init__(self):
        self.trials = Trials()
        
    
    def score(self, params):
        """
        The score function to use in hyperopt.
        It is average of the validation error of xgboost models, using a K-fold validation

        df_train and kf (k-fold) need to be defined previously
        """
        
        if self.output_file != None:
            with open(self.output_file, "a") as myfile:
                try:
                    myfile.write(str(self.trials.losses()[-2])+'\n')
                except IndexError:
                    print 'Index error'
                myfile.write(str(params)+', ')

        print "Training with params : "
        print params
        num_round = int(params['n_estimators'])
        del params['n_estimators']

        score = 0.
        for train_index, valid_index in self.kf:

            df_train = self.df_train.iloc[train_index]
            df_valid = self.df_train.iloc[valid_index]

            # fit the model
            self.fit(df_train, self.features, self.target, params, num_round)

            # results of the model on validation data
            predictions = self.predict(df_valid[self.features])

            # computing the accuracy of predictited similar pictures
            accuracy = np.mean(df_valid[self.target].values == np.round(predictions))
            print 'accuracy:', accuracy
            score -= accuracy/float(len(self.kf))
            
            #score -= roc_auc_score(df_valid[self.target].values, predictions)

        print "\tScore {0}\n\n".format(score)
        return {'loss': score, 'status': STATUS_OK}



    def optimize(self, space, df_train, features, target, kf, output_file=None):
        """
        Find the best hyperparameters in the parameters-space, that minimize the score function.
        
        INPUTS:
        
        space    : (dict) space of paramters to explore
        df_train : (pd.DataFrame) contains the training data: the features and the target
        features : (string list) list of column names to use as features in the model
        target   : (string) name of the column to use as target
        kf       : (sklearn.cross_validation object) a KFold object for cross validation
        """
        
        self.df_train = df_train
        self.features = features
        self.target = target
        self.kf = kf
        self.output_file = output_file

        # searching for the best parameters:
        self.best = fmin(self.score, space, algo=tpe.suggest, trials=self.trials, max_evals=250)

        print best
        
        best = self.best.copy()
        numrounds = int(best['n_estimators'])
        del best['n_estimators']
        
        # fit an Xgboost model with the best parameters on the whole training data
        self.fit(df_train, features, target, best, numrounds)
              
        
        
    def fit(self, df, features, target, params, numrounds):
        """
        Fit an Xgboost model with parameters defined in params and numrounds
        """
        X_train = df[features]
        y_train = df[target]
        dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
        self.model = xgb.train(params, dtrain,  numrounds )

    def predict(self, df):
        """
        return the output of the model for data in df
        """
        data = xgb.DMatrix(df.values)
        return self.model.predict(data)
