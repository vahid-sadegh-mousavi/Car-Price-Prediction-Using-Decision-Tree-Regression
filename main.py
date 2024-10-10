import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Car details v3.csv', header=0)

# Rename columns for consistency
df.rename(columns = {'selling_price':'price', 'mileage':'consumption', 'max_power':'power', 'seller_type':'seller'}, inplace=True)

# Extract company name from 'name' column
companyname = df['name'].apply(lambda x : x.split(' ')[0])
df.insert(1, "companyname", companyname)
df.drop(['name'], axis=1, inplace=True)

# Drop duplicates
df = df.drop_duplicates()

# Replace company names with regions
def replace_name(a,b):
    df.companyname.replace(a,b,inplace=True)

company_replacements = {
    'Toyota': 'japan', 'Nissan': 'japan', 'Datsun': 'japan', 'Mitsubishi': 'japan', 'Isuzu': 'japan', 'Lexus': 'japan',
    'Maruti': 'india', 'Mahindra': 'india', 'Ashok': 'india', 'Force': 'india', 'Tata': 'india', 'Ambassador': 'india',
    'Honda': 'japan', 'Hyundai': 'korea', 'Kia': 'korea', 'Daewoo': 'korea',
    'Ford': 'usa', 'Chevrolet': 'usa', 'Jeep': 'usa', 'Renault': 'france', 'Peugeot': 'france',
    'Volkswagen': 'germany', 'BMW': 'germany', 'Mercedes-Benz': 'germany', 'Audi': 'germany', 'Opel': 'germany',
    'Fiat': 'euro', 'Volvo': 'euro', 'Jaguar': 'euro', 'MG': 'euro', 'Land': 'euro', 'Skoda': 'euro'
}

for k, v in company_replacements.items():
    replace_name(k, v)

df.rename(columns={'companyname':'companyhome'}, inplace=True)

# Drop 'torque' column, unnecessary for this project
df.drop('torque', axis='columns', inplace=True)

# Convert consumption and engine data to numeric
df['consumption'] = df['consumption'].apply(lambda x: float(str(x).split()[0]) if pd.notnull(x) else x)
df['engine'] = df['engine'].apply(lambda x: float(str(x).split()[0]) if pd.notnull(x) else x)
df['engine'] = df['engine'].replace(".0", "")

# Prepare the dataset for machine learning
df = df[['price', 'companyhome', 'year', 'engine', 'power', 'seats', 'km_driven', 'fuel', 'seller', 'transmission', 'owner', 'consumption']]

# Handle missing values
mv = df.consumption.mean()
df.consumption = df.consumption.fillna(mv).replace(0, mv)

mv = df.engine.mean()
df.engine = df.engine.fillna(mv)

mv = df.power.mean()
df.power = df.power.fillna(mv).replace(0, mv)

mv = df.seats.mean()
df.seats = df.seats.fillna(mv).astype(int)

# Train-test split
x = df.loc[:, df.columns != 'price']
y = df['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Decision tree regression
regtree = tree.DecisionTreeRegressor(max_depth=3)
regtree.fit(x_train, y_train)

# Predictions
y_train_pred = regtree.predict(x_train)
y_test_pred = regtree.predict(x_test)

# Model performance
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Test R2:", r2_score(y_test, y_test_pred))
print("Train R2:", r2_score(y_train, y_train_pred))

# Visualization
sns.jointplot(x='year', y='price', data=df)
plt.show()