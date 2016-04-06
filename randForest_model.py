from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp, fmin, STATUS_OK, Trials

class RandFModel():
    
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
                    pass
                myfile.write(str(params)+', ')
                   
                             
        print "Training with params : "
        print params
        n_estimators = int(params['n_estimators'])
        del params['n_estimators']
                             
        score = 0.
        for train_index, valid_index in self.kf:

            df_train = self.df_train.iloc[train_index]
            df_valid = self.df_train.iloc[valid_index]
            
            model = RandomForestClassifier(n_estimators=n_estimators, **params)

            # fit the model
            model.fit(df_train[self.features], df_train[self.target])
            
            # computing the accuracy of predictited similar pictures
            accuracy = model.score(df_valid[self.features], df_valid[self.target])
            print 'accuracy:', accuracy
            score -= accuracy/float(len(self.kf))

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

        print 'best parameters:', self.best
        
        if self.output_file != None:
            with open(self.output_file, "a") as myfile:
                myfile.write(str(self.trials.losses()[-2])+'\n')
                myfile.write('best_parameters, '+str(self.best))
        
        best = self.best.copy()
        n_estimators = int(best['n_estimators'])
        del best['n_estimators']
        
        # fit a RandomForest model with the best parameters on the whole training data
        self.model = RandomForestClassifier(n_estimators=n_estimators, **best)
        self.model.fit(df_train[self.features], df_train[self.target])
              

    def predict(self, df):
        """
        return the output of the model for data in df
        """
        return self.model.predict(df)
