from geopy.distance import vincenty

class KNearestNeighbors(object):

    def __init__(self, k=5, distance=euclidean_distance):
        self.k = k
        self.distance = distance

    def fit(self, X, y):
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

        for i, values in enumerate(top_k):

            result[i] = Counter(values).most_common(1)[0][0]

        return result
