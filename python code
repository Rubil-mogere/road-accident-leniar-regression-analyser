import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import pickle
import itertools

# Load the accident dataset from the uploaded CSV file
data = pd.read_csv('accident.csv')

# Get unique values for categorical variables
categorical_columns = ['Weather', 'Road_Type', 'Time_of_Day', 'Vehicle_Type']
unique_values = {col: data[col].unique() for col in categorical_columns}

# Create a DataFrame for all possible combinations of categorical values
new_data = pd.DataFrame(list(itertools.product(*unique_values.values())), columns=categorical_columns)

# Encode categorical variables using one-hot encoding
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_data = encoder.fit_transform(new_data)

# Manually construct the column names for the one-hot encoded variables
column_names = []
for category, categories in zip(categorical_columns, encoder.categories_):
    for cat in categories:
        column_names.append(f"{category}_{cat}")

encoded_data = pd.DataFrame(encoded_data, columns=column_names)

# Drop the original categorical columns
new_data = new_data.drop(categorical_columns, axis=1)

# Concatenate the encoded data with the original data
new_data = pd.concat([new_data, encoded_data], axis=1)

# Separate independent and dependent variables
independent_variables = new_data
dependent_variable = data['Accident_Severity']

# Create and fit the linear regression model
model = LinearRegression()
model.fit(independent_variables, dependent_variable)

# Save the model and encoder for future use
with open('accident_severity_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('encoder.pkl', 'wb') as encoder_file:
    pickle.dump(encoder, encoder_file)

# Predict accident severity for a hypothetical set of independent variables
new_data_sample = new_data.sample(1)  # Choose a random combination
predicted_severity = model.predict(new_data_sample)[0]
print("Predicted accident severity:", predicted_severity)
