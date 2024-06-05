
# Analyse des Ventes Walmart

## Introduction

Bienvenue dans ton projet d'analyse des ventes de Walmart ! Dans ce projet, tu vas explorer les données de ventes hebdomadaires de 45 magasins Walmart répartis à travers divers départements. L'objectif est de découvrir des insights précieux, d'identifier les tendances clés et de construire des modèles prédictifs pour mieux comprendre les performances des magasins.

## Contexte

Walmart, l'une des plus grandes chaînes de distribution au monde, génère des quantités massives de données de ventes chaque semaine. Analyser ces données peut révéler des informations cruciales pour la gestion des stocks, les stratégies de marketing et l'optimisation des ventes. En tant que data analyst junior, tu as été chargé de ce projet fascinant pour aider Walmart à prendre des décisions éclairées basées sur les données.

## Objectifs du Projet

1. **Exploration des Données**
    - Comprendre la structure des données.
    - Effectuer une analyse descriptive pour obtenir des statistiques clés.
    - Visualiser les ventes hebdomadaires pour chaque magasin et chaque département.

2. **Modélisation de Régression Linéaire et Ridge/Lasso**
    - Préparer les données pour la modélisation.
    - Construire et évaluer des modèles de régression linéaire, Ridge et Lasso pour prédire les ventes hebdomadaires.
    - Visualiser les performances des modèles à travers les ventes réelles et prédites.

3. **Arbre de Décision pour la Performance des Magasins**
    - Créer une variable cible pour catégoriser les performances des magasins.
    - Entraîner un modèle d'arbre de décision pour prédire les performances des magasins.
    - Évaluer et visualiser l'arbre de décision.

## Méthodologie

### 1. Exploration des Données

Nous avons commencé par charger les données et effectuer une conversion des dates pour faciliter l'analyse temporelle. Ensuite, nous avons réalisé une analyse statistique pour résumer les ventes hebdomadaires :

###python
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
print(df.describe())
Statistiques Descriptives

Pour visualiser les ventes hebdomadaires par magasin et par département, nous avons utilisé des boxplots :

python
Copier le code
sns.boxplot(x='Store', y='Weekly_Sales', data=df)
sns.boxplot(x='Dept', y='Weekly_Sales', data=df)
Ventes Hebdomadaires par Magasin

2. Modélisation de Régression Linéaire, Ridge et Lasso
Nous avons préparé nos ensembles d'entraînement et de test, puis construit des modèles de régression linéaire, Ridge et Lasso :

python
Copier le code
models=[{'name':'LinearRegression','model':LinearRegression()},
        {'name':'Ridge','model': Ridge()},{'name':'Lasso','model': Lasso()}
        ]

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
Les résultats ont été évalués à l'aide de la MSE et du R², et visualisés comme suit :

3. Arbre de Décision pour la Performance des Magasins
Nous avons ensuite créé une nouvelle variable 'Performance' pour catégoriser les performances des magasins et entraîné un modèle d'arbre de décision :

python
Copier le code
df['Performance'] = np.where(df['Weekly_Sales'] > df['Weekly_Sales'].median(), 1, 0)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
Arbre de Décision

Résultats

Visualisation des Ventes Mensuelles Totales
En analysant les ventes mensuelles totales, nous avons pu identifier les mois avec des pics de ventes significatifs.

python
Copier le code
def graphique_par_an(x):
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df.set_index('Date', inplace=True)
    start_date = pd.Timestamp(f'{x}-01-01')
    end_date = pd.Timestamp(f'{x}-12-31')
    df_filtered = df[(df.index >= start_date) & (df.index < end_date)]
    df_monthly_sales = df_filtered['Weekly_Sales'].resample('M').sum()
    df_monthly_sales_million = df_monthly_sales / 1e6
    plt.figure(figsize=(14, 7))
    plt.plot(df_monthly_sales.index, df_monthly_sales_million)
    plt.ticklabel_format(style='plain', axis='y')
    plt.title('Ventes mensuelles totales')
    plt.xlabel('Date')
    plt.ylabel('Ventes mensuelles en millions')
    plt.grid(True)
    plt.show()

graphique_par_an(2011)
Performance des Modèles
La régression linéaire, Ridge et Lasso ont montré une capacité modérée à prédire les ventes hebdomadaires, tandis que l'arbre de décision a bien classé les magasins selon leur performance.

Conclusion

Ce projet a permis de mettre en évidence l'importance de l'analyse des données pour comprendre et prévoir les tendances de ventes. Grâce à des techniques de visualisation et de modélisation, nous avons pu extraire des insights précieux et proposer des modèles prédictifs robustes.

Instructions

Prérequis
Python 3.6 ou supérieur
Packages listés dans requirements.txt
Installation
Clone le dépôt :


watara13
