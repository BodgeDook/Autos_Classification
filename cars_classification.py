import data_downloading, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.preprocessing import StandardScaler

'''
*Some Information about the learning data:*

Price: a number
Fuel Type: Diesel/Petrol/Hybrid (one-hot encoding)
Transmission: Automatic/Manual (one-hot encoding)
Engine volume (in cc): a number (usually, 4-length)
[bhp, rpm for bhp, Nm, rpm for Nm]: numbers *(bhp and Nm are floats/doubles)*
Drivetrain: AWD/RWD/FWD => [*('full' drive/'back' drive/'fore' drive)*] (one-hot encoding)
Length, Width, Height: numbers

!We use all of the generations of these makes' models!
The amount of learning and testing examples all together: ~120 models for each brand
'''

makes_dict = {1 : 'BMW', 2 : 'Honda', 3 : 'Mercedes-Benz', 4 : 'Toyota'}
targets = data_downloading.targets_arr
data = data_downloading.data_arr

scaler = StandardScaler()
data = scaler.fit_transform(data) # scaling the parameters equally so that there is no high scatter!

X_train, X_test, y_train, y_test = train_test_split(
    data, targets, random_state = 0)

print()
print(f'X_train size is {X_train.shape}')
print(f'y_train size is {y_train.shape}')
print()
print(f'X_test size is {X_test.shape}')
print(f'y_test size is {y_test.shape}')
print()

# ======================================================================== #
# ================= An example of Neural Network's work: ================= #
# ======================================================================== #
X_new_One_dict = pd.DataFrame({
    'Price': [1_550_000],
    'Fuel Type': ['Petrol'],
    'Transmission': ['Automatic'],
    'Engine (in cc)': [1496],
    'bhp': [110.0],
    'rpm for bhp': [6000],
    'Nm': [140.0],
    'rpm for Nm': [4400],
    'Drivetrain': ['FWD'],
    'Length': [4215],
    'Width': [1760],
    'Height': [1630]
}) # Toyota Corolla Rumion (2006 - 2017)

X_new_One_encoded = pd.get_dummies(X_new_One_dict, columns = ['Fuel Type', 'Transmission', 'Drivetrain'])

missing_cols = set(data_downloading.rest_data.columns) - set(X_new_One_encoded.columns)
for i in missing_cols:
    X_new_One_encoded[i] = 0

X_new_One_encoded = X_new_One_encoded[data_downloading.rest_data.columns]
X_new_One_final = X_new_One_encoded.to_numpy()
X_new_One_final = scaler.transform(X_new_One_final)
# ======================================================================== #
# ======================================================================== #
# ======================================================================== #

# knn = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski') # making a knn-object for our neural network
knn = KNC(n_neighbors = 7, weights = 'distance', algorithm = 'auto', leaf_size = 30,
          p = 1, metric = 'minkowski', metric_params = None, n_jobs = None)
knn.fit(X_train, y_train) # main training

print('Learning complete!')
print(f'Correctness of the test data: {knn.score(X_test, y_test):.2f}') # the amount of correctness
prediction = (knn.predict(X_new_One_final))[0]
print(f'Your prediction was: {makes_dict[prediction]}')