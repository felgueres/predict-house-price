import pandas as pd
from geopy.distance import great_circle as geodistance
import numpy as np
from tools import MRAE

class Preprocessing(object):
    '''
    Preprocess data to avoid time-leakage.

    Parameters
    ----------
    data: path
        Path to file with data

    Output
    ------
    Returns feature matrix and house pricing (ie. Preprocessing.x, Preprocessing.y)
    '''

    def __init__(self, path2data):

        # read dataset to pandas dataframe
        self.df = pd.read_csv(path2data)

        # Create a temporary dataframe for home i be used to avoid time leakage.
        self.df_home_i = None

        # Create temporary pointer to home_i
        self.home_i = None

    def _datatypes(self):
        '''
        Convert datatypes suitable for manipulation.
        '''
        #Convert to DatetimeIndex
        self.df.close_date = pd.DatetimeIndex(self.df.close_date)

        return self

    def _time_leakage(self, home_i):
        '''
        Filters dataframe to specified home h_i
        such a home j should be considered a neighbor to home i
        only if the close date of j occurred prior to the close date of i.

        Parameters
        ----------
        home_i: integer
            Index of a home in the dataframe.

        Output
        ------
        None (Dataframe for home_i was modified).

        '''
        #Index homes.
        self.home_i = home_i

        #Mask to avoid time leakage.
        self.df_home_i = self.df.ix[self.df.close_date > self.df.iloc[self.home_i].close_date].copy()

    def _distance(self):
        '''
        Compute distances between home_i and neighbors.

        Parameters
        ----------
        home_i: integer
            Index of a home in the dataframe.

        Output
        ------
        None (Dataframe for home_i was modified).

        '''
        #Index Home i
        home_i = self.df.iloc[self.home_i]

        #Get home_i coordenates
        home_i_coord = (home_i.latitude, home_i.longitude)

        #Create lat_lon feature for simple reference
        self.df_home_i['lat_lon'] = zip(self.df_home_i.latitude, self.df_home_i.longitude)

        #Apply function to dataframe element-wise
        self.df_home_i['dist2home_i_miles'] = self.df_home_i.lat_lon.apply(lambda (coord2): geodistance(home_i_coord,coord2).miles)

    def get_training_data(self, home_i):
        '''
        Parameters
        ----------
        home_i: integer
            Index number of home to predict value for.

        Output
        ------
        X: array-like
            Returns feature matrix.

        Y: array-like
            Returns home closing price.
        '''
        #Convert to DatetimeIndex
        self._datatypes()

        #Filter dataframe avoiding time leakage for home i.
        self._time_leakage(home_i)

        #Compute distances from home i to all remaining homes.
        self._distance()

        #Get closing price as y.
        y = self.df_home_i.pop('close_price').values

        #Get Distance as X.
        X = self.df_home_i.pop('dist2home_i_miles').values

        return X, y

class KNearestNeighbors(object):
    '''
    KNN regressor to calculate house pricing.

    Parameters
    ----------

    k: integer
        Number of k nearest neighbors.

    distance: function
        Function to calculate distance (not neccesarily spatial).
    '''

    def __init__(self, k=4, distance=geodistance, weights = None):
        self.k = k
        self.distance = distance
        self.Preprocessing = None #This is to import the Preprocessing class.
        self.predictions = [] #To store model predictions.
        self.weights = None #If we were to make this model more complex, suitable weigths would be selected based on cross-validates feature importance.
        self.model_performance = None #Attribute to get MRAE.

    def fit(self, Preprocessing):
        '''
        Takes in Preprocessing class to fit feature matrix (X) and closing price (y), assuring there is not time leakage.
        '''
        #Loads data and proprocessing class.
        self.Preprocessing = Preprocessing

    def predict(self):
        '''
        Predict home price.
        '''
        #For each home, iterate to get X_train, y_train and predict k neighrest with weights w.
        #Note the Preprocessing class handles calculating the distance.

        #Create y_pred, y_actual results.
        self.predictions = []

        #Truncate dataframe to only first 1000 predictions.
        for home_i in self.Preprocessing.df.index.tolist()[:1000]:

            #Get training data
            X_train, y_train = self.Preprocessing.get_training_data(home_i)

            #Sort and get top k.
            top_k = y_train[X_train.argsort()[:self.k]]  #sort and take top k

            #Use weights to compute house pricing.
            #In this case I will use the mean, ie. for k = 4, w = 0.25
            home_i_pred = np.mean(np.absolute(top_k)) #I noted about 4% of the dataset has negative sign values for pricing, I'll consider those to be positive for business sense.

            #Get actual home price.
            home_i_actual = np.absolute(self.Preprocessing.df.iloc[home_i].close_price)

            #Append to list as tuple for later conversion to numpy array.
            self.predictions.append((home_i_pred, home_i_actual))

        #Truncate df to only the first 1000 predictions.
        self.df_truncated = self.Preprocessing.df.iloc[:1000,:].copy()

        #Insert Column with predictions
        self.df_truncated['pred'] = np.array(self.predictions)[:,0] #from above, note the position for predictions in the tuple is 0.

    def performance(self):
        '''
        Compute performance of model element wise and overall.
        '''
        #Compute RAE element-wise.
        self.df_truncated['RAE'] = np.absolute(np.absolute(self.df_truncated.close_price) - self.df_truncated.pred) / np.absolute(self.df_truncated.close_price)
        #Compute MRAE.
        self.model_performance = np.median(self.df_truncated.RAE)

        #Return Model Performance
        print 'Model Median Relative Absolute Error: {0:.1%}'.format(self.model_performance)

if __name__ == '__main__':

    #Initialize model.
    knn = KNearestNeighbors(k=4, distance = geodistance)

    #Initialize Preprocessing class with data.
    data = Preprocessing('../data/data.csv')

    #Fit dataset to model using the Preprocessing class.
    knn.fit(data)

    #Predict
    knn.predict()

    #Model Performance
    knn.performance()
