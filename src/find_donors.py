#######################################################################
#                                                                     #
#                    IMPORTS                                          #
#                                                                     #
#######################################################################  


from IPython.display import display           # display(df) prints large DataFrames in nice table format.
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as ps
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import sys
from typing import Dict, List, Union
from time import time


#######################################################################
#                                                                     #
#                    DATALOADER                                       #
#                                                                     #
#######################################################################  


class DataLoader(object):
    """Class which initializes a dataset from a csv file."""
    
    def __init__(self, col_targets=None):
        
        # Column which contains target labels
        self.col_targets = col_targets
    
    def load(self, file_path: str):
        """Create dataset from csv file. """
        
        df = pd.read_csv(file_path)
        
        # Perform feature / target split
        # if target column was defined
        
        if self.col_targets in df.columns:            

            df_features = df.drop(self.col_targets, axis=1)
            s_targets = df[self.col_targets]
            
            return df_features, s_targets        
               
        else:
            
            return df   


#######################################################################
#                                                                     #
#                    DATA PREPROCESSOR                                #
#                                                                     #
#######################################################################  
        
        
class DataPreprocessor(object):
    
    def __init__(self,
                 log_features: Union[List, None] = None,
                 log_offsets: Union[List, None] = None,
                 norm_features: Union[List, None] = None,
                 map_targets: Union[Dict, None] = None):
        
        # Features which will be log transformed
        self.log_features = log_features
        
        # Offsets to apply when doing log transforms
        self.log_offsets = log_offsets    
        
        # Features which will be normalized
        self.norm_features = norm_features
        
        # Dictionnary for integer encoding targets
        self.map_targets = map_targets
        
    def log_transform(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Applies logarithmic tranformation to features."""
        
        df_out = df_in.copy()
        
        n = len(self.log_features)
        
        for i in range(n):
            
            feature = self.log_features[i]
            offset = self.log_offsets[i]
            
            df_out[feature] = df_out[feature].apply(lambda x: np.log(x+offset))        
        
        return df_out

    def normalize(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features."""
        
        df_out = df_in.copy()
        
        fts = self.norm_features
        
        if fts is not None:
            
            if len(fts) > 0:
            
                scaler = MinMaxScaler() # By default, scales from 0 to 1
                df_out[fts] = scaler.fit_transform(df_out[fts])
         
        return df_out
    
    
    def one_hot_encode(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical values."""
        
        df_out = df_in.copy()
        
        df_out = pd.get_dummies(df_out)        
        
        return df_out        
    
    
    def integer_encode(self, s_in: pd.Series) -> pd.Series:
        """Integer encode categorical values."""
        
        s_out = s_in.copy()
        
        s_out = s_out.map(self.map_targets)
        
        return s_out
    
    def preprocess_features(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Preprocess DataFrame of features."""
        
        df_out = df_in.copy()
        
        df_out = self.log_transform(df_out)
        df_out = self.normalize(df_out)
        df_out = self.one_hot_encode(df_out)
        
        return df_out
    
    def preprocess_targets(self, s_in: pd.Series) -> pd.Series:
        """Preprocess Series of targets."""
        
        s_out = s_in.copy()
        
        s_out = self.integer_encode(s_out)
        
        return s_out        
        
        
        
#######################################################################
#                                                                     #
#                    VISUALIZATION UTILITIES                          #
#                                                                     #
#######################################################################  


def print_df_unique_values(df: pd.DataFrame, nbins=None):
    """Print DataFrame unique values info for each column."""

    for col in df.columns:
        
        s = df[col]
    
        # Column name
    
        print()
        print("COLUMN NAME")
        print()
        print(s.name)
        
        # Unique values
        
        print()
        print("UNIQUE VALUES")
        print()
        uniques = s.unique()
        uniques = np.sort(uniques)
        print()
        print(uniques)
        
        # Unique values count
        
        print()
        print("UNIQUE VALUES COUNT")
        print()
        print(s.value_counts().sort_index())
        
        # Print histogram
        
        print()
        print("UNIQUE VALUES COUNT HISTOGRAM")
        print()
        fig = px.histogram(x=s, nbins=nbins)
        fig.layout.xaxis.title = s.name
        fig.layout.height= 400
        fig.layout.width = 800
        fig.show()


def print_pre_feature(s_raw: pd.Series, s_pre: pd.Series, nbinsx=None, ymin=None, ymax=None) -> None:
    """Print numerial feature information, before/after pre-processing."""
    
    series = [s_raw, s_pre]
    
    n_cols = len(series)
    
    xaxis_titles = ["Before", "After"]
    
    fig = ps.make_subplots(rows=1, cols=n_cols, shared_yaxes=True)
    
    for i in range(n_cols):
        
        s = series[i]

        trace = go.Histogram(x=s, nbinsx=nbinsx)
        fig.add_trace(trace, row=1, col=i+1)
        
        fig.update_xaxes(title_text = xaxis_titles[i], row=1, col=i+1)
        fig.update_yaxes(title_text = "count", row=1, col=i+1)   

        if (ymin is not None) and (ymax is not None):
            fig.update_yaxes(range = [ymin, ymax], row=1, col=i+1)
            fig.update_yaxes(title_text = "count <br> (truncated at " + str(ymax) + " )", row=1, col=i+1)
    
    fig.layout.title.text = s_raw.name
    fig.show()


#######################################################################
#                                                                     #
#                    TRAINING UTILITIES                               #
#                                                                     #
#######################################################################  


def run_grid_search(model, hyperparameters, scorer, features, labels):
    
    # Initialize grid search object

    gs = GridSearchCV(model, hyperparameters, scoring=scorer)

    gs = gs.fit(features, labels)
    
    return gs


def print_gs_for_report(gs: GridSearchCV) -> None:
    """Print grid search results in nice format for this project report."""

    df_cv_results = pd.DataFrame(gs.cv_results_)

    # Drop columns not used in final report.
    columns_to_drop = []
    columns_to_drop.extend(["mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time", "params"])
    columns_to_drop.extend(["split0_test_score", "split1_test_score", "split2_test_score", "split3_test_score", "split4_test_score"])
    columns_to_drop.extend(["std_test_score"])
    df_cv_results = df_cv_results.drop(columns_to_drop, axis=1)

    # Add column with name of model.    
    model_name = gs.best_estimator_.__class__.__name__
    df_cv_results.insert(loc=0, column="model", value = model_name)

    # Add "note" column which flags the best model.
    df_cv_results["note"] = ""
    index_best_score = np.argmin(df_cv_results["rank_test_score"])
    df_cv_results.loc[index_best_score, "note"] = "best"

    print()
    print(df_cv_results.to_markdown())
    print()
    
    
#######################################################################
#                                                                     #
#                    MAIN                                             #
#                                                                     #
#######################################################################  
    
def run_pipeline(argv):

    # Input / Output parameters

    input_file = argv[0]
    output_file = "output.csv"

    # Data Pre-processor parameters

    log_features = ["capital-gain", "capital-loss"]
    log_offsets = [1,1]
    numerical_features = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
   
    # Machine learning model parameters

    model_file = "model.pkl"

    # Data Loader

    dl = DataLoader()
    df = dl.load(input_file)

    # Data Pre-Processor

    dpp = DataPreprocessor(log_features, log_offsets, numerical_features)
    df = dpp.preprocess_features(df)

    # Machine Learning Model

    with open(model_file, "rb") as f:

        model = pickle.load(f)

    # Fill any missing column with 0's
    # Columns can go missing during one-hot encoding if
    # a columns in input.csv does not contain
    # all the values which existed during training).

    features = model.feature_names_in_

    for ft in features:
        if ft not in df.columns:
            df[ft] = 0

    # Reorder columns of df_features in same order as during training
                
    df = df[features]
   
    # Predict

    predictions = model.predict(df)
    # Output results

    s = pd.Series(predictions)
    s.to_csv(output_file, index=False, header=False)
    

if __name__ == "__main__":

    argv = sys.argv[1:]
    run_pipeline(argv)







    
