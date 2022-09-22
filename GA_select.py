# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
import datetime
import os

'''GA based approach'''
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn_genetic.callbacks import ProgressBar
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


class GA:
    def __init__(self, dataset, target, drop):
       self.dataset = dataset
       self.target =  target
       self.drop = drop
        
    def data_process(dataset, target, drop):
        '''data and labels'''
        Label=dataset[target]
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(Label)
        
        if drop is not None: 
            additional_columns= list(drop)
            X = dataset.drop(columns=additional_columns)
            X = X.drop(columns=[target])
        else:
            X = dataset.drop(columns= [target])

        return X, y, additional_columns

    def split_transform_data(X, y):
        quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

        X_train_trans = quantile_transformer.fit_transform(X_train)
        X_test_trans = quantile_transformer.transform(X_test)

        '''convert back to dataframes'''
        X_train_trans = pd.DataFrame(X_train_trans,columns=X.columns)
        X_test_trans = pd.DataFrame(X_test_trans,columns=X.columns)

        clf = GradientBoostingClassifier(n_estimators=10)
        clf.fit(X_train_trans, y_train)
        y_predict = clf.predict(X_test_trans)
        accuracy_no_GA = accuracy_score(y_test, y_predict)
        print("accuracy score without GA selection: ", "{:.2}".format(accuracy_no_GA))
        return clf, X_train_trans, X_test_trans, y_test, y_train, accuracy_no_GA 

  
    def main(generations, population_size, crossover_probability, 
                max_features, outdir, clf, X_train_trans, X_test_trans, 
                y_test, y_train, hr_start_time):
        print("Running main function")

        cv_results, history, selected_features, plot = GA.run_GA(generations=generations,
                                    population_size=population_size,
                                    crossover_probability=crossover_probability,
                                    max_features=max_features,
                                    clf = clf, X_train_trans = X_train_trans,
                                    X_test_trans = X_test_trans, 
                                    y_test = y_test,
                                    y_train = y_train)

        results_df= pd.DataFrame(cv_results)
        results_df.to_csv('results_df.csv')

        history_df= pd.DataFrame(history)
        history_df.to_csv('history_df.csv')

        plt.figure()
        sns.violinplot(data=history_df.iloc[:,1:])
        plt.savefig('history_results.png')

        pd.DataFrame(selected_features).to_csv(str(hr_start_time) +'_selected_features.csv')
        return history_df, selected_features, plot

    def run_GA(generations,population_size,crossover_probability,max_features,
        clf, X_train_trans, X_test_trans, y_test, y_train):

        evolved_estimator = GAFeatureSelectionCV(
            estimator=clf,
            cv=5,
            scoring="accuracy",
            population_size=population_size,
            generations=generations,
            n_jobs=-1,
            crossover_probability=crossover_probability,
            mutation_probability=0.05,
            verbose=True,
            max_features= max_features,
            keep_top_k=3,
            elitism=True
            )

        callback = ProgressBar()
        evolved_estimator.fit(X_train_trans, y_train, callbacks=callback)
        features = evolved_estimator.best_features_
        y_predict_ga = evolved_estimator.predict(X_test_trans.iloc[:,features])
        accuracy = accuracy_score(y_test, y_predict_ga)
        print(evolved_estimator.best_features_)
        
        plt.figure()
        plot = plot_fitness_evolution(evolved_estimator, metric="fitness")
        plt.savefig('fitness.png')


        selected_features= list(X_test_trans.iloc[:,features].columns)
        cv_results= evolved_estimator.cv_results_
        history= evolved_estimator.history
        return cv_results, history, selected_features, plot
    
    def running (generations, population_size, crossover_probability, 
                    max_features, outdir, clf, X_train_trans, X_test_trans,
                    y_test, y_train, accuracy_no_GA, additional_columns):  
        generations = int(generations)
        population_size = int(population_size)
        crossover_probability =float(crossover_probability)

        if max_features is not None:
            print (f'max_features has been set (value is {max_features})')
            max_features = int(max_features)
        else:
            max_features = None
            print (f'max_features has not been set (value is  max_features)')

        start_time = time.time()
        hr_start_time = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H-%M-%S')
        
        RESULTS_DIR = outdir
        LOG_FILE = os.path.join(RESULTS_DIR , f'log.txt')
        logging.basicConfig(format='%(levelname)s:%(message)s',
                            level=logging.INFO,
                            handlers=[logging.FileHandler(LOG_FILE),
                            logging.StreamHandler()])
        
        logging.info(f"starting time: {hr_start_time}")


        history_df, selected_features, plot = GA.main(generations, population_size, crossover_probability, 
                    max_features, outdir, clf, X_train_trans, X_test_trans, y_test, y_train, hr_start_time)


        TOTAL_TIME = f'Total time required: {time.time() - start_time} seconds'
        hr_end_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H-%M-%S')
        logging.info(f"Number of generations: {generations}")
        logging.info(f"Number of features max allowed: {max_features}")
        logging.info(f"Population_size: { population_size}")
        logging.info(f"Crossover_probability: {crossover_probability}")
        logging.info(f"Max accuracy with all features: {accuracy_no_GA}")
        logging.info(f"End time: {hr_end_time}")
        logging.info(f"Max fitness with selection: {history_df.fitness.max()}")
        logging.info(f"Selected features:, {selected_features}")
        logging.info(f"Removed features by user:, {additional_columns}")

        logging.info(TOTAL_TIME)
        logging.info('done!')
        return hr_start_time, plot, selected_features
