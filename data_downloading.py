import pandas as pd

file = 'vehicles_dataset(v1.6.7)_copy.xlsx'
def autos_data_reading(dataset_file):
    return pd.read_excel(dataset_file)

def autos_data_reforming():
    autos_data = autos_data_reading(file)
    brands_dict = {'BMW': 1, 'Honda': 2, 'Mercedes-Benz': 3, 'Toyota': 4}
    autos_data['Brand'] = autos_data['Brand'].map(brands_dict) # "replacing" brands' names with their encodes
    return autos_data

def targets_forming():
    autos_data = autos_data_reforming()
    targets_arr = autos_data['Brand'].to_numpy()
    return targets_arr

def rest_data_forming():
    autos_data = autos_data_reforming()
    # one-hot encoding for string information:
    autos_data = pd.get_dummies(autos_data, columns = ['Fuel Type', 'Transmission', 'Drivetrain'])
    rest_data = autos_data.drop('Brand', axis = 1)
    return rest_data

rest_data = rest_data_forming()

data_arr = rest_data[['Price', 'Engine (in cc)', 'bhp', 'rpm for bhp', 'Nm', 'rpm for Nm', 'Length',
                      'Width', 'Height'] + list(rest_data.columns[-8:])].to_numpy() # data!
targets_arr = targets_forming() # targets!