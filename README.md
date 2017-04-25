## Opendoor: Data Science Takehome Assignment
## Pablo Felgueres
## 4/24/2017

## Questions:  

### 1) Using the dataset provided, please build a k-NN model for k = 4 that avoids time leakage.

### _See folder /src with two files:_

- model.py: Contains two classes to build the kNN model.

  Preprocessing class:

  - Initializes with path to raw data and loads it to Pandas dataframe
  - Methods of class:

    - **_datatypes**: Converts close_date datatype from string to DatetimeIndex object for manipulation.

    - **_time_leakage**: Filters dataframe of given home _home i_ such a
                        _home j_ should be considered a neighbor to _home i_ only if
                        the _close date of j_ occurred prior to the _close date of i_.

    - **_distance**: Computes distance between coordinates of given home _home_i_ and all other homes.

    - **_get_training_data**: Returns X_train (Distance2house_i) and y_train (house_price) for a specified _home_i_.

  KNearestNeighbors class:

  - Initializes with k number and type of distance to compare similarity of elements.
  - Methods of class:

    - **fit**: Takes in Preprocessing class to fit feature matrix (X) and closing price (y),
                assuring there is no time leakage.

    - **predict**: Computes prediction for each _house_i_ given the computed distances and specified weights.
                  In this case the distance is purely spatial and I'm setting the weights to be equally distributed for each neighbor (ex. k = 4, w = 0.25, which is the expected value/mean).

    - **performance**: Computes the performance of model using median relative absolute error as metric.

- tools.py: Contains auxiliary functions.

### Instructions to model

    > 1) Navigate to folder _/src_

    > 2) Run in terminal -ipython model.py

    > 3) It will print out the model performance metric.

    > 4) Explore the kNN class for documentation on getting predictions and preprocessed data to avoid time-leaks.

### 2) What is the performance of the model measured in Median Relative Absolute Error?

  The data pipeline I built in this allotted timeframe is not optimized for computation efficiency (details on Q.5).
  I truncated the computations after 1000 predictions which accounts for only 1.1% of the dataset.

  The Median Relative Absolute Error for the current model on the truncated sample is:

  - 18.7%

### 3) What would be an appropriate methodology to determine the optimal k?

  Assuming the optimal k is what provides the lowest performance error of the model; to determine k I would cross-validate using range values of k, plotting the error and selecting a value which generalizes the lowest error.

  Equally important, prior knowledge on this business/market could provide other heuristics to consider when cross-validating and selecting the optimal k.

### 4) Do you notice any spatial or temporal trends in error?

  By performing a quick scatter matrix on the 1000 sample predictions, I notice the performance of the model reduces dramatically around a lat, lon = (36, -100). However, I also notice those coordinates represent the bulk of the sample so it's not conclusive whether that is a error trend in the model.

  ![alt text](https://github.com/felgueres/opendoor/blob/master/images/scatter_matrix.JPG)

  The time-span of my sample predictions covers a 6-month period during 2014. There seems to be an intra-month seasonality but before jumping into a conclusion I would want to perform further predictions; then inspect the error distribution, fit to a function for quantitative inference.

### 5) How would you improve this model?

Note the only feature I'm considering to compute similarity between homes is geographic distance.
My model also currently assumes a uniform contribution per data point in the local neighborhood, the next step would be to cross-validate whether uniform is indeed better than weighting proportionally to the distance.

With this limited dataset I would also introduce a temporal feature with the hypothesis that the closer the closing dates between two homes, the more correlated their closing price would be since they share temporal market conditions.
Then check if the hypothesis is true, and based on the temporal contribution to the housing price, adjust the weights to improve the model (via cross-validation as well).

On the other hand, considering the value of homes is determined by characteristics other than location and temporality, the most logical step would be to understand the key features that describe and correlate to the value of a home. For example, we could do this via quantitative analysis (ex. linear regression regularized L1) as well as to obtain intangibles/heuristics from field agents.

In terms of which model or function to use, an interesting consideration is whether interpretability is of value or not. Most likely we could use a non-parametric model (ex. Random Forest) and achieve the higher accuracies; however, in Opendoor's particular interest, a client is interested in transparency and would presumably value the information taken into account as well as the weights to achieve the final pricing so one must navigate a trade-off between interpretability and accuracy.

### 6) How would you productionize this model?

The model I built is very slow at the moment for various reasons:

- kNN is all about saving data into memory. In this case, I'm doing a computationally expensive procedure since I'm copying a distinct dataset for every single home and recurrently calculating distances with the time leakage constraint (this constraint is actually beneficial to the overall time calculation since it reduces row count or N).
My approach is currently a 'brute force', which without the time leakage constraint would be O[N^2], which is fairly unreasonable even with one feature.

- A step to the first point would be to partition the feature space so for every new prediction, the amount of computations is reduced given a data point coincides with a partition. If we keep the dimensions of the feature space low, a possibility would be to build a K-Dimensional tree to infer distances between homes as opposed to doing computations repeatedly.

-  Another idea with the current dataset would be to implement a k-Means clustering technique to assign geographical centroids to associate homes before computing distances.

- Finally, before production we would want to test the cross-validated model with out-of-sample unseen data.
