import pandas as pd

dataset_file = 'vehicles_dataset(v1.6.7)_copy.xlsx'
autos_data = pd.read_excel(dataset_file)

brands_dict = {'BMW' : 1, 'Honda' : 2, 'Mercedes-Benz' : 3, 'Toyota' : 4}
autos_data['Brand'] = autos_data['Brand'].map(brands_dict) # "replacing" brands' names with their encodes
# one-hot encoding for string information:
autos_data = pd.get_dummies(autos_data, columns = ['Fuel Type', 'Transmission', 'Drivetrain'])

targets_arr = autos_data['Brand'].to_numpy() # targets!
rest_data = autos_data.drop('Brand', axis = 1)
data_arr = rest_data[['Price', 'Engine (in cc)', 'bhp', 'rpm for bhp', 'Nm', 'rpm for Nm', 'Length', 'Width', 'Height']\
                     + list(rest_data.columns[-8:])].to_numpy() # data!