import numpy as np


def MRAE(y_true, y_pred):
    '''
    Computes median relative absolute error for a model.

    Parameters
    ----------

    y_true: array-like
        Actual closing selling prices.

    y_pred: array-like
        Predicted closing selling prices

    '''

    return np.median(np.absolute(y_pred - y_true) / y_true)

if __name__ == '__main__':
    pass
