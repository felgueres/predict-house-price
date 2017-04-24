import pandas as pd
from geopy.distance import vincenty as geodistance
import numpy as np


class preprocessing(object):
    '''
    Preprocess data to avoid time-leakage.

    Parameters
    ----------
    data: path
        Path to file with data

    Output
    ------
    Returns feature matrix and house pricing (ie. preprocessing.x, preprocessing.y)
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
        None (Only dataframe for home_i was modified.

        '''
        #Index homes.
        self.home_i = home_i

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
        None

        '''
        #Index Home i
        home_i = self.df_home_i.iloc[self.home_i]

        #Get home_i coordenates
        home_i_coord = (home_i.latitude, home_i.longitude)

        #Create lat_lon feature for simple reference
        self.df_home_i['lat_lon'] = zip(self.df_home_i.latitude, self.df_home_i.longitude)

        #Apply function to dataframe element-wise
        self.df_home_i['dist2home_i_miles'] = self.df_home_i.lat_lon.apply(lambda (coord2): geodistance(home_i_coord,coord2).miles)

    def get_training_data(self, home_i):
        '''
        Output
        ------
        X: array-like
            Returns feature matrix.

        Y: array-like
            Returns home closing price.
        '''

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

    def __init__(self, k=4, distance=geodistance):
        self.k = k
        self.distance = distance
        self.preprocessing = None #This is to import the preprocessing class.
        self.predictions = [] #To store model predictions.

    def fit(self, preprocessing):
        '''
        Takes in preprocessing class to fit feature matrix (X) and closing price (y), assuring there is not time leakage.
        '''
        #Loads data and proprocessing class.
        self.preprocessing = preprocessing

    def predict(self):
        '''
        Predict method, retur
        '''
        #For each home, iterate to get X_train, y_train and predict k neighrest with weights w.
        #Note the preprocessing class handles calculating the distance.

        #Create y_pred, y_actual results.
        self.predictions = []

        for home_i in self.preprocessing.df.index.tolist()[:1]:

            #Get training data
            X_train, y_train = self.preprocessing.get_training_data(home_i)

            #Sort and get top k.
            top_k = y_train[X_train.argsort()[:self.k]]  #sort and take top k

            #Use weights to compute house pricing.
            #In this case I will use the mean, ie. for k = 4, w = 0.25
            home_i_price = np.mean(np.absolute(top_k)) #I noted about 4% of the dataset has negative sign values for pricing, I'll consider those to be positive for business sense.

            self.predictions.append((y_train, home_i_price))

        return self.predictions

    def performance(self):
        '''
        Compute performance of model.
        '''
        #Need to expand the predictions and use metric from tools.



if __name__ == '__main__':

    #Initialize model.
    knn = KNearestNeighbors(k=4, distance = geodistance)

    #Initialize preprocessing class with data.
    data = preprocessing('../data/data.csv')

    #Fit dataset to model using the preprocessing class.
    knn.fit(data)

    #Predict
    knn.predict()
