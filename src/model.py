from geopy.distance import vincenty


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

    def __init__(self, data = '../data/data.csv'):

        # read dataset to pandas dataframe
        self.df = pd.read_csv(data)

        # Create a temporary dataframe to be used to avoid time leakage.
        self.df_temp = None

    def _datatypes(self):
        '''
        Convert datatypes suitable for manipulation.
        '''
        self.df.close_date = pd.DatetimeIndex(self.df.close_date)

        return self

    def time_leakage(self, i):
        '''
        
        '''




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

    def __init__(self, k=5, distance=vincenty):
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        '''
        Fits feature matrix (X) and closing price (y), assuring there is not time leakage.
        '''
        self.X_train = X
        self.y_train = y

    def predict(self, X):

        X = X.reshape( (-1, self.X_train.shape[1]) )

        distances = np.zeros((X.shape[0], self.X_train.shape[0]))

        for i, x in enumerate(X):

            for j, x_train in enumerate(self.X_train):

                distances[i, j] = self.distance(x_train, x)

        top_k = y[distances.argsort()[:,:self.k]]  #sort and take top k

        result = np.zeros(X.shape[0])

        # for i, values in enumerate(top_k):
        #
        #     result[i] = Counter(values).most_common(1)[0][0]

        return result

if __name__ == '__main__':

    # Running the model.

    pass
