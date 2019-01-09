# Add import statements
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
# Load the data
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")

# Make and fit the linear regression model
# Fit the model and Assign it to bmi_life_model
model = LinearRegression()

# Features: Country, Life expectancy, BMI
X = np.array(bmi_life_data[['BMI']])
y = np.array(bmi_life_data['Life expectancy'])

bmi_life_model = model.fit(X, y)

# Make a prediction using the model
# Predict life expectancy for a BMI value of 21.07931
laos_life_exp = model.predict([21.07931])
