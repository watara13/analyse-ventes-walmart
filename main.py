import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import plot_tree
def graphique_par_an (x):
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df.set_index('Date', inplace=True)
    start_date = pd.Timestamp(f'{x}-01-01')
    end_date = pd.Timestamp(f'{x }-12-12')
    df_filtered = df[(df.index >= start_date) & (df.index < end_date)]
    df_monthly_sales = df_filtered['Weekly_Sales'].resample('ME').sum()
    df_monthly_sales_million = df_monthly_sales / 1e6
    plt.figure(figsize=(14, 7))
    plt.plot(df_monthly_sales.index, df_monthly_sales_million)
    plt.ticklabel_format(style='plain', axis='y')
    plt.title('Ventes mensuelles totales')
    plt.xlabel('Date')
    plt.ylabel('Ventes mensuelles en millions')
    plt.grid(True)
    plt.show()


models=[{'name':'LinearRegression','model':LinearRegression()},
        {'name':'Ridge','model': Ridge()},{'name':'Laso','model': Lasso()}
        ]

url ='walmart-sales-dataset-of-45stores.csv'
df = pd.read_csv(url)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

print(df.info())

print("Premières lignes du dataframe :")
print(df.head())

print("\nInformations sur le dataframe :")
print(df.info())

print("\nStatistiques descriptives :")
print(df.describe())

print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())


print("Statistiques descriptives des ventes hebdomadaires :")
print(df['Weekly_Sales'].describe().apply(lambda x: format(x, 'f')))
graphique_par_an(2011)
plt.figure(figsize=(10, 6))
plt.hist(df['Weekly_Sales'], bins=50, color='blue', edgecolor='black')
plt.ticklabel_format(style='plain', axis='x')
plt.ticklabel_format(style='plain', axis='y')
plt.title('Distribution des ventes hebdomadaires')
plt.xlabel('Ventes hebdomadaires')
plt.ylabel('Fréquence')
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 8))
sns.boxplot(x='Store', y='Weekly_Sales', data=df)
plt.ticklabel_format(style='plain', axis='y')
plt.title('Ventes hebdomadaires par magasin')
plt.xlabel('Magasin')
plt.ylabel('Ventes hebdomadaires')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


X = df[['Store', 'Month', 'Year']]
y = df['Weekly_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
for item in models:
    name=item['name']
    model=item['model']
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'{name} :Mean Squared Error: {mse}')
    print(f'{name} :R^2 Score: {r2}')


plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.5})
plt.ticklabel_format(style='plain', axis='x')
plt.ticklabel_format(style='plain', axis='y')
plt.xlabel('Ventes Réelles')
plt.ylabel('Ventes Prédites')
plt.title('Régression Linéaire - Ventes Réelles vs Ventes Prédites')
plt.grid(True)
plt.show()

df['Performance'] = np.where(df['Weekly_Sales'] > df['Weekly_Sales'].median(), 1, 0)

features = ['Store', 'Month', 'Year']
target = 'Performance'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure(figsize=(20, 10))
plt.ticklabel_format(style='plain', axis='x')
plt.ticklabel_format(style='plain', axis='y')
plot_tree(model, feature_names=features, class_names=['Low', 'High'], filled=True)
plt.title('Arbre de Décision - Performance des Magasins')
plt.show()