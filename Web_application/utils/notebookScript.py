# %%
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


current_folder = os.getcwd()
base_dir = os.path.dirname(current_folder)
print (current_folder)
print(base_dir)



# %% [markdown]
# ### Chargement des données
# 

# %%
marketing_data = pd.read_csv(current_folder +"/generated_data/marketing_dataset.csv")
customers_data = pd.read_csv(current_folder +"/generated_data/customers_dataset.csv")
products_data = pd.read_csv(current_folder +"/generated_data/products_dataset.csv")
sales_data = pd.read_csv(current_folder +"/generated_data/sales_dataset.csv")
print (customers_data.shape)
print (marketing_data.shape)
print (products_data.shape)
print (sales_data.shape)

# %%
marketing_data.isnull().sum()

# %%
marketing_data

# %%
customers_data

# %%
sales_data

# %%
products_data

# %%
#marketing_data.isnull().sum()
customers_data.info()

# %% [markdown]
# ### Enjeux Marketing
# 

# %%
marketing_data1 = marketing_data.drop(columns=['Campaign_ID'], axis=1)
#changement de format de date 
marketing_data1['Start_Date'] = pd.to_datetime(marketing_data1['Start_Date'])
marketing_data1['End_Date'] = pd.to_datetime(marketing_data1['End_Date'])

# Création de la colonne 'Campaign_Duration' en jours
marketing_data1['Campaign_Duration'] = (marketing_data1['End_Date'] - marketing_data1['Start_Date']).dt.days

marketing_data1


# %%
# Exemple de KPI pour la SWOT(Strengths,Weaknesses,Opportunities,Threats)
marketing_data1['CTR'] = marketing_data1['Clicks'] / marketing_data1['Impressions']  # Impressions:visibilité
marketing_data1['Conversion_Rate'] = marketing_data1['Conversions'] / marketing_data1['Clicks'] #conversion :Nombre d’actions attendues
marketing_data1['CPC'] = marketing_data1['Budget'] / marketing_data1['Clicks']  # Coût par clic
marketing_data1['CPA'] = marketing_data1['Budget'] / marketing_data1['Conversions']  # Coût par acquisition

# KPI quotidiens
marketing_data1['Budget_per_Day'] = marketing_data1['Budget'] / marketing_data1['Campaign_Duration']
marketing_data1['Conversions_per_Day'] = marketing_data1['Conversions'] / marketing_data1['Campaign_Duration']
marketing_data1['Clicks_per_Day'] = marketing_data1['Clicks'] / marketing_data1['Campaign_Duration']
marketing_data1['Impressions_per_Day'] = marketing_data1['Impressions'] / marketing_data1['Campaign_Duration']


# Calcul moyen par canal
summary_by_channel = marketing_data1.groupby('Channel')[
['CTR', 'Conversion_Rate', 'CPC', 'CPA','Campaign_Duration']
].mean().sort_values(
by=['CTR','Conversion_Rate','CPC','CPA'],
ascending=[True,True,True,True])
summary_by_channel

# %%
# Recherche de CTR max et CPA max
channel_max_CTR = summary_by_channel['CTR'].idxmax()
print("Données du Channel avec max CTR :")
print(summary_by_channel.loc[channel_max_CTR])

print("\nDonnées du Channel avec max CPA :")
print(summary_by_channel.loc[channel_max_CTR])


# %%
#Signification des indicateurs:
summary_by_channel.plot(kind='bar', figsize=(12, 6), width=1)

# %%
summary_by_channel_per_day = marketing_data1.groupby('Channel')[
['Budget_per_Day','Conversions_per_Day','Clicks_per_Day','Impressions_per_Day']
].mean().sort_values(
by=['Budget_per_Day','Conversions_per_Day','Clicks_per_Day','Impressions_per_Day'],
ascending=[True,True,True,True])

# %%
summary_by_channel_per_day.plot(kind='bar', figsize=(12, 6), width=0.6)

# %%
customers_data

# %%
products_data

# %%
products_data.drop('Product_ID',axis=1)

# %%
sales_data

# %%
sales_data.shape

# %%
products_data.shape

# %%
# Fusion des csv
merged_data = pd.merge(sales_data,products_data, on="Product_ID", how="left")
merged_data

# %%
merged_data.isnull().sum ()

# %%
concat_data = pd.concat([customers_data,   merged_data], axis=1)
concat_dat = concat_data.drop(columns=['Product_ID','Customer_ID'],axis=1)
concat_dat

# %%
data = concat_dat.copy()
data = data.drop(columns=['Date','Product_Name','Sale_Price','Sale_ID'],axis=1)
data

# %% [markdown]
# ### Exploration des données ensemble
# 

# %%
df = data[['Name','Age','Gender','Location','Join_Date','Total_Spent','Quantity','Channel','Category','Brand']].copy()
df_Avg_Price = df['Total_Spent'] / df['Quantity']
df['Avg_Price'] = df_Avg_Price
df


# %%


# %%
df.isnull().sum()

# %%
df.info()

# %% [markdown]
# ### Division de df en deux sous groupes
# 

# %%
data_cat = []
data_num = []
for i,c in zip(df.dtypes,df.columns):
    if i == object:
        data_cat.append(c)
    else:
        data_num.append(c)
data_cat = df[data_cat]
data_num = df[data_num]
		

# %%
data_num

# %%
data_cat

# %% [markdown]
# ### Data pour la segmentation des clients
# 

# %%
df

# %%
# 1. Colonnes numériques et catégorielles
num_features = ['Age', 'Total_Spent', 'Quantity','Avg_Price']
cat_features = ['Gender','Location','Channel', 'Category', 'Brand']

# 2. Préprocesseurs
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')

# 3. Pipeline de prétraitement
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# 4. Pipeline complet avec KMeans (k = 4)
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('clustering', KMeans(n_clusters=5, random_state=42))
])

# 5. Chargement des données
df = df.copy()

# 6. Appliquer le pipeline
pipeline.fit(df)

# 7. Ajouter les clusters au DataFrame
df['Cluster'] = pipeline.named_steps['clustering'].labels_

# 8. Visualiser
print(df[['Name', 'Age', 'Total_Spent', 'Cluster']].head(10))


# %%
df['Cluster'].value_counts(ascending=True)

# %%
# Extraire les données après prétraitement
X_processed = pipeline.named_steps['preprocessing'].transform(df)

# Réduire en 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# Tracer
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Cluster'], cmap='viridis')
plt.title('Segmentation des clients avec KMeans')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()


# %% [markdown]
# ### Resultat pour la segmentation des clients
# 

# %%
df['Cluster'].value_counts(ascending=True)

# %% [markdown]
# ### Groupe_0
# 

# %%
# Filtrer les clients du cluster 0
clients_cluster_0 = df[df['Cluster'] == 0]['Name'].unique()
print(clients_cluster_0)
#print("La somme des clients dans le groupe 0 est ",clients_cluster_0.count())
clients_cluster_0_avg_age = df[df['Cluster'] == 0]['Age'].mean()
print(f"La moyenne d'age pour le groupe 0 est :{clients_cluster_0_avg_age} Ans")
clients_cluster_0_avg_Total_Spent = df[df['Cluster'] == 0]['Total_Spent'].mean()
print(f"La moyenne de dépense pour le groupe 0 est: {clients_cluster_0_avg_Total_Spent} $")
clients_cluster_0_max_Total_Spent = df[df['Cluster'] == 0]['Total_Spent'].max()
print(f"La dépense maximale pour le groupe 0 est: {clients_cluster_0_max_Total_Spent} $")

# %% [markdown]
# ### Groupe_1
# 

# %%
# Filtrer les clients du cluster 1
clients_cluster_1 = df[df['Cluster'] == 1]['Name'].unique()
print(clients_cluster_1)
#print("La somme des clients dans le groupe 0 est ",clients_cluster_0.count())
clients_cluster_1_avg_age = df[df['Cluster'] == 1]['Age'].mean()
print(f"La moyenne d'age pour le groupe 1 est :{clients_cluster_1_avg_age} Ans")
clients_cluster_1_avg_Total_Spent = df[df['Cluster'] == 1]['Total_Spent'].mean()
print(f"La moyenne de dépense pour le groupe 1 est: {clients_cluster_1_avg_Total_Spent} $")
clients_cluster_1_max_Total_Spent = df[df['Cluster'] == 1]['Total_Spent'].max()
print(f"La dépense maximale pour le groupe 1 est: {clients_cluster_1_max_Total_Spent} $")

# %% [markdown]
# ### Groupe_2
# 

# %%
# Filtrer les clients du cluster 2
clients_cluster_2 = df[df['Cluster'] == 2]['Name'].unique()
print(clients_cluster_2)
clients_cluster_2_avg_age = df[df['Cluster'] == 2]['Age'].mean()
print(f"La moyenne d'age pour le groupe 2 est :{clients_cluster_2_avg_age} Ans")
clients_cluster_2_avg_Total_Spent = df[df['Cluster'] == 2]['Total_Spent'].mean()
print(f"La moyenne de dépense pour le groupe 2 est: {clients_cluster_2_avg_Total_Spent} $")
clients_cluster_2_max_Total_Spent = df[df['Cluster'] == 2]['Total_Spent'].max()
print(f"La dépense maximale pour le groupe 2 est: {clients_cluster_2_max_Total_Spent} $")

# %% [markdown]
# ### Groupe_3
# 

# %%
# Filtrer les clients du cluster 3
clients_cluster_3 = df[df['Cluster'] == 3]['Name'].unique()
print(clients_cluster_3)
clients_cluster_3_avg_age = df[df['Cluster'] == 3]['Age'].mean()
print(f"La moyenne d'age pour le groupe 3 est :{clients_cluster_3_avg_age} Ans")
clients_cluster_3_avg_Total_Spent = df[df['Cluster'] == 3]['Total_Spent'].mean()
print(f"La moyenne de dépense pour le groupe 3 est: {clients_cluster_3_avg_Total_Spent} $")
clients_cluster_3_max_Total_Spent = df[df['Cluster'] == 3]['Total_Spent'].max()
print(f"La dépense maximale pour le groupe 3 est: {clients_cluster_3_max_Total_Spent} $")

# %% [markdown]
# ### Groupe_4
# 

# %%
# Filtrer les clients du cluster 4
clients_cluster_4 = df[df['Cluster'] == 4]['Name'].unique()
print(clients_cluster_4)
clients_cluster_4_avg_age = df[df['Cluster'] == 4]['Age'].mean()
print(f"La moyenne d'age pour le groupe 4 est :{clients_cluster_4_avg_age} Ans")
clients_cluster_4_avg_Total_Spent = df[df['Cluster'] == 4]['Total_Spent'].mean()
print(f"La moyenne de dépense pour le groupe 4 est: {clients_cluster_4_avg_Total_Spent} $")
clients_cluster_4_max_Total_Spent = df[df['Cluster'] == 4]['Total_Spent'].max()
print(f"La dépense maximale pour le groupe 4 est: {clients_cluster_4_max_Total_Spent} $")

# %%
df

# %%
# df[['Total_Spent','Quantity']]

# %% [markdown]
# #### Modelisation preprocessing
# 

# %%
df['fidelity_score_raw'] = (df['Total_Spent'] * 0.7) + (df['Quantity'] * 100 * 0.3)
df

# %%
#Seuil à définir (ex: médiane ou top 30%)
threshold = df['fidelity_score_raw'].quantile(0.7)  # Top 30% des scores
df['Loyal_Customer'] = (df['fidelity_score_raw'] >= threshold).astype(int)
df

# %%
# 2. Sélection des features
features = df[['Age', 'Gender','Total_Spent','Quantity','Channel', 'Category','Cluster','Avg_Price','Loyal_Customer']]
features

# %% [markdown]
# ### save clean data
# 

# %%
data_dir = os.path.join(current_folder, 'data')
print (data_dir)
filename = f"cleaned_data.csv"
features.to_csv(os.path.join(data_dir, filename), index=False)


