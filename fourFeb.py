import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Housing Price Dataset.csv - Housing (1).csv')
df.head()

df['furnishingstatus'].nunique()
columns_to_transform = ['mainroad', 'guestroom', 'basement','hotwaterheating','airconditioning','prefarea']
df[columns_to_transform] = df[columns_to_transform].replace({'yes': 1, 'no': 0})
 
df['furnishingstatus'] = df['furnishingstatus'].replace({'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2})
corr_matrix = df.corr()

plt.figure(figsize=(10, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

df.hist(figsize=(10, 10), bins=10)
plt.suptitle("Histograms for All Columns", fontsize=16)
plt.show()

X = df.drop('price', axis=1)
y = df['price']

print(X.shape)
print(y.shape)
print(X [:10])
print (y [:10])

from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler()
sc = ['price', 'area']
df[sc] = scaler.fit_transform(df[sc])

df.drop("colmun", axis=1)
from sklearn.model_selection import train_test_split
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LinearRegression
 
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)
print(lr_y_pred)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, lr_y_pred)
mae = mean_absolute_error(y_test,lr_y_pred)
r2 = r2_score(y_test, lr_y_pred)
 
print("\nModel Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-Squared (R2): {r2:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, lr_y_pred, color='blue', alpha=0.7, label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
 
plt.xlabel('Target Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression Predicted vs Target Values')
plt.legend()
plt.grid(True)
 
plt.show()

print('Training score',lr_model.score(X_train,y_train))
print('Testing score',lr_model.score(X_test,y_test))
