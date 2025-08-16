import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')


class CustomerSegmentationComparator:
    """
    Classe pour comparer différents modèles de segmentation client
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=random_state)
        
    def prepare_data(self, df, features):
        """
        Prépare les données pour la segmentation
        
        Args:
            df (pd.DataFrame): DataFrame des clients
            features (list): Liste des features à utiliser
        
        Returns:
            np.array: Données normalisées
        """
        # Sélectionner et nettoyer les features
        X = df[features].copy()
        
        # Gérer les valeurs manquantes
        X = X.fillna(X.median())
        
        # Normalisation
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, X
    
    def fit_kmeans(self, X, n_clusters=5):
        """Applique K-Means clustering"""
        model = KMeans(
            n_clusters=n_clusters, 
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        labels = model.fit_predict(X)
        
        self.models['KMeans'] = model
        self.results['KMeans'] = {
            'labels': labels,
            'centers': model.cluster_centers_,
            'inertia': model.inertia_
        }
        
        return labels
    
    def fit_gmm(self, X, n_components=5):
        """Applique Gaussian Mixture Model"""
        model = GaussianMixture(
            n_components=n_components,
            random_state=self.random_state,
            covariance_type='full'
        )
        model.fit(X)
        labels = model.predict(X)
        probabilities = model.predict_proba(X)
        
        self.models['GMM'] = model
        self.results['GMM'] = {
            'labels': labels,
            'probabilities': probabilities,
            'means': model.means_,
            'aic': model.aic(X),
            'bic': model.bic(X)
        }
        
        return labels
    
    def fit_dbscan(self, X, eps=0.5, min_samples=5):
        """Applique DBSCAN clustering"""
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        
        self.models['DBSCAN'] = model
        self.results['DBSCAN'] = {
            'labels': labels,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'n_outliers': np.sum(labels == -1)
        }
        
        return labels
    
    def fit_hierarchical(self, X, n_clusters=5):
        """Applique Agglomerative Clustering"""
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        labels = model.fit_predict(X)
        
        self.models['Hierarchical'] = model
        self.results['Hierarchical'] = {
            'labels': labels,
            'n_clusters': n_clusters
        }
        
        return labels
    
    def calculate_metrics(self, X, labels):
        """
        Calcule les métriques d'évaluation des clusters
        
        Args:
            X: Données d'entrée
            labels: Étiquettes des clusters
            
        Returns:
            dict: Dictionnaire des métriques
        """
        # Vérifier qu'il y a plus d'un cluster
        if len(set(labels)) < 2:
            return {'silhouette': -1, 'calinski_harabasz': -1, 'davies_bouldin': 100}
        
        # Exclure les outliers pour DBSCAN
        if -1 in labels:
            mask = labels != -1
            X_clean = X[mask]
            labels_clean = labels[mask]
            
            if len(set(labels_clean)) < 2:
                return {'silhouette': -1, 'calinski_harabasz': -1, 'davies_bouldin': 100}
        else:
            X_clean = X
            labels_clean = labels
        
        try:
            metrics = {
                'silhouette': silhouette_score(X_clean, labels_clean),
                'calinski_harabasz': calinski_harabasz_score(X_clean, labels_clean),
                'davies_bouldin': davies_bouldin_score(X_clean, labels_clean)
            }
        except:
            metrics = {'silhouette': -1, 'calinski_harabasz': -1, 'davies_bouldin': 100}
            
        return metrics
    
    def fit_all_models(self, X, n_clusters=5):
        """
        Applique tous les modèles de clustering
        
        Args:
            X: Données d'entrée
            n_clusters: Nombre de clusters souhaité
        """
        print("🔄 Application des modèles de clustering...")
        
        # K-Means
        print("   - K-Means...")
        self.fit_kmeans(X, n_clusters)
        
        # Gaussian Mixture Model
        print("   - Gaussian Mixture Model...")
        self.fit_gmm(X, n_clusters)
        
        # DBSCAN
        print("   - DBSCAN...")
        self.fit_dbscan(X, eps=0.3, min_samples=10)
        
        # Hierarchical Clustering
        print("   - Hierarchical Clustering...")
        self.fit_hierarchical(X, n_clusters)
        
        print("✅ Tous les modèles appliqués!")
    
    def evaluate_models(self, X):
        """
        Évalue tous les modèles avec différentes métriques
        
        Args:
            X: Données d'entrée
            
        Returns:
            pd.DataFrame: Résultats de l'évaluation
        """
        evaluation_results = []
        
        for model_name, result in self.results.items():
            labels = result['labels']
            metrics = self.calculate_metrics(X, labels)
            
            evaluation_results.append({
                'Model': model_name,
                'N_Clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'Silhouette_Score': round(metrics['silhouette'], 4),
                'Calinski_Harabasz': round(metrics['calinski_harabasz'], 2),
                'Davies_Bouldin': round(metrics['davies_bouldin'], 4),
                'Outliers': np.sum(labels == -1) if -1 in labels else 0
            })
        
        return pd.DataFrame(evaluation_results)
    
    def visualize_results(self, X, original_features=None):
        """
        Visualise les résultats de tous les modèles
        
        Args:
            X: Données d'entrée
            original_features: Features originales pour les noms d'axes
        """
        # Réduction de dimensionnalité pour visualisation
        if X.shape[1] > 2:
            X_pca = self.pca.fit_transform(X)
        else:
            X_pca = X
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        colors = ['purple', 'yellow', 'lightblue', 'lightgreen', 'orange', 'red', 'brown', 'pink']
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            ax = axes[idx]
            labels = result['labels']
            
            # Créer un scatter plot
            unique_labels = set(labels)
            for i, label in enumerate(unique_labels):
                if label == -1:  # Outliers pour DBSCAN
                    ax.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], 
                             c='black', marker='x', s=50, alpha=0.6, label='Outliers')
                else:
                    color = colors[i % len(colors)]
                    ax.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], 
                             c=color, alpha=0.7, s=30, label=f'Cluster {label}')
            
            # Ajouter les centres pour K-Means
            if model_name == 'KMeans' and 'centers' in result:
                if result['centers'].shape[1] > 2:
                    centers_pca = self.pca.transform(result['centers'])
                else:
                    centers_pca = result['centers']
                ax.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                          c='red', marker='x', s=200, linewidths=3, label='Centers')
            
            ax.set_title(f'{model_name}\n({len(set(labels)) - (1 if -1 in labels else 0)} clusters)', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('PCA 1' if X.shape[1] > 2 else 'Feature 1')
            ax.set_ylabel('PCA 2' if X.shape[1] > 2 else 'Feature 2')
        
        plt.tight_layout()
        plt.suptitle('Comparison of Clustering Models', fontsize=16, fontweight='bold', y=1.02)
        plt.show()
    
    def get_best_model(self, X):
        """
        Détermine le meilleur modèle basé sur les métriques
        
        Args:
            X: Données d'entrée
            
        Returns:
            str: Nom du meilleur modèle
        """
        evaluation_df = self.evaluate_models(X)
        
        # Score composite (plus le score est élevé, mieux c'est)
        # Silhouette: plus élevé = mieux
        # Calinski-Harabasz: plus élevé = mieux  
        # Davies-Bouldin: plus bas = mieux
        
        evaluation_df['Composite_Score'] = (
            evaluation_df['Silhouette_Score'] * 0.4 +
            (evaluation_df['Calinski_Harabasz'] / evaluation_df['Calinski_Harabasz'].max()) * 0.4 +
            (1 - evaluation_df['Davies_Bouldin'] / evaluation_df['Davies_Bouldin'].max()) * 0.2
        )
        
        best_model = evaluation_df.loc[evaluation_df['Composite_Score'].idxmax(), 'Model']
        return best_model, evaluation_df
    
    def get_cluster_profiles(self, df, features, model_name='KMeans'):
        """
        Génère les profils des clusters
        
        Args:
            df: DataFrame original
            features: Liste des features utilisées
            model_name: Nom du modèle à utiliser
            
        Returns:
            pd.DataFrame: Profils des clusters
        """
        if model_name not in self.results:
            print(f"Modèle {model_name} non trouvé!")
            return None
        
        labels = self.results[model_name]['labels']
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = labels
        
        # Calculer les moyennes par cluster
        cluster_profiles = df_with_clusters.groupby('Cluster')[features].agg(['mean', 'count']).round(2)
        
        return cluster_profiles


def analyze_customer_segmentation(data_path, features_list, n_clusters=5, 
                                customer_id_col='Customer_ID', save_results=False):
    """
    Analyse complète de segmentation client avec vos données
    
    Args:
        data_path (str): Chemin vers votre fichier CSV
        features_list (list): Liste des colonnes à utiliser pour la segmentation
        n_clusters (int): Nombre de clusters souhaité
        customer_id_col (str): Nom de la colonne Customer_ID
        save_results (bool): Sauvegarder les résultats
        
    Returns:
        tuple: (segmentator, df_with_clusters, evaluation_results)
    """
    
    print("🔄 CHARGEMENT ET ANALYSE DE VOS DONNÉES")
    print("="*50)
    
    try:
        # Charger vos données
        df = pd.read_csv(data_path)
        print(f"✅ Données chargées: {df.shape}")
        print(f"📊 Colonnes disponibles: {list(df.columns)}")
        
        # Vérifier que les features existent
        missing_features = [f for f in features_list if f not in df.columns]
        if missing_features:
            print(f"❌ Features manquantes: {missing_features}")
            print(f"📋 Features disponibles: {[col for col in df.columns if col != customer_id_col]}")
            return None, None, None
        
        print(f"🎯 Features sélectionnées: {features_list}")
        print(f"📈 Aperçu des données:")
        print(df[features_list].describe())
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None, None, None
    
    # Initialiser le comparateur
    segmentator = CustomerSegmentationComparator()
    
    # Préparer les données
    print(f"\n🔧 PRÉPARATION DES DONNÉES")
    print("-" * 30)
    
    X_scaled, X_original = segmentator.prepare_data(df, features_list)
    print(f"✅ Données normalisées: {X_scaled.shape}")
    
    # Appliquer tous les modèles
    print(f"\n🤖 APPLICATION DES MODÈLES DE CLUSTERING")
    print("-" * 45)
    segmentator.fit_all_models(X_scaled, n_clusters=n_clusters)
    
    # Évaluer les modèles
    print(f"\n📊 ÉVALUATION DES MODÈLES")
    print("-" * 30)
    evaluation_df = segmentator.evaluate_models(X_scaled)
    print(evaluation_df.to_string(index=False))
    
    # Trouver le meilleur modèle
    best_model, detailed_results = segmentator.get_best_model(X_scaled)
    print(f"\n🏆 MEILLEUR MODÈLE: {best_model}")
    print(f"📈 Score composite le plus élevé!")
    
    # Ajouter les clusters au DataFrame original
    best_labels = segmentator.results[best_model]['labels']
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = best_labels
    df_with_clusters['Best_Model'] = best_model
    
    # Visualiser les résultats
    print(f"\n📈 VISUALISATION DES RÉSULTATS")
    print("-" * 35)
    segmentator.visualize_results(X_scaled, features_list)
    
    # Profils détaillés des clusters
    print(f"\n👥 PROFILS DÉTAILLÉS DES CLUSTERS ({best_model})")
    print("-" * 50)
    profiles = segmentator.get_cluster_profiles(df, features_list, best_model)
    if profiles is not None:
        print(profiles)
        
        # Statistiques par cluster
        print(f"\n📊 RÉPARTITION DES CLUSTERS:")
        cluster_counts = pd.Series(best_labels).value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            if cluster_id != -1:  # Ignorer les outliers
                percentage = (count / len(best_labels)) * 100
                print(f"   Cluster {cluster_id}: {count} clients ({percentage:.1f}%)")
        
        if -1 in best_labels:  # Si il y a des outliers (DBSCAN)
            outliers = np.sum(best_labels == -1)
            percentage = (outliers / len(best_labels)) * 100
            print(f"   Outliers: {outliers} clients ({percentage:.1f}%)")
    
    # Sauvegarder les résultats si demandé
    if save_results:
        print(f"\n💾 SAUVEGARDE DES RÉSULTATS")
        print("-" * 30)
        
        # Sauvegarder le DataFrame avec clusters
        output_file = data_path.replace('.csv', '_with_clusters.csv')
        df_with_clusters.to_csv(output_file, index=False)
        print(f"✅ Données avec clusters sauvegardées: {output_file}")
        
        # Sauvegarder l'évaluation des modèles
        eval_file = data_path.replace('.csv', '_model_evaluation.csv')
        detailed_results.to_csv(eval_file, index=False)
        print(f"✅ Évaluation des modèles sauvegardée: {eval_file}")
        
        # Sauvegarder les profils des clusters
        if profiles is not None:
            profiles_file = data_path.replace('.csv', '_cluster_profiles.csv')
            profiles.to_csv(profiles_file)
            print(f"✅ Profils des clusters sauvegardés: {profiles_file}")
    
    print(f"\n✨ ANALYSE TERMINÉE!")
    print("="*50)
    
    return segmentator, df_with_clusters, evaluation_df


def quick_segmentation_analysis(data_path, features_list, n_clusters=5):
    """
    Version rapide pour tester rapidement la segmentation
    
    Args:
        data_path (str): Chemin vers vos données
        features_list (list): Features pour la segmentation
        n_clusters (int): Nombre de clusters
    """
    
    results = analyze_customer_segmentation(
        data_path=data_path,
        features_list=features_list,
        n_clusters=n_clusters,
        save_results=True
    )
    
    if results[0] is not None:
        print(f"\n🎯 RECOMMANDATIONS:")
        print("-" * 20)
        print(f"✅ Utilisez le modèle: {results[2].loc[results[2]['Composite_Score'].idxmax(), 'Model']}")
        print(f"📊 Nombre optimal de clusters trouvés")
        print(f"💾 Résultats sauvegardés automatiquement")
        print(f"📈 Visualisations générées")
    
    return results


# Configuration basée sur votre code de ML customer loyalty
def analyze_customer_loyalty_data():
    """
    Analyse de segmentation configurée pour vos données de fidélité client
    Basée sur votre script process.py
    """
    
    print("🎯 ANALYSE DE SEGMENTATION - DONNÉES DE FIDÉLITÉ CLIENT")
    print("="*60)
    
    # CONFIGURATION BASÉE SUR VOTRE CODE ORIGINAL
    # Chemin vers vos données (depuis votre script process.py)
    data_path = "data/cleaned_data.csv"
    
    # Features courantes dans les données de fidélité client
    # Adaptez selon les colonnes réelles de votre dataset
    loyalty_features = [
        'Age',                    # Âge du client
        'Total_Spent',           # Montant total dépensé
        'Purchase_Frequency',    # Fréquence d'achat
        'Days_Since_Last_Purchase',  # Récence (jours depuis dernier achat)
        'Average_Order_Value',   # Panier moyen
        'Tenure_Days'           # Ancienneté en jours
    ]
    
    # Features alternatives si les noms diffèrent
    alternative_features = [
        'customer_age',
        'total_amount_spent', 
        'number_of_purchases',
        'recency_days',
        'avg_purchase_amount',
        'customer_lifetime_days'
    ]
    
    print(f"📁 Fichier de données: {data_path}")
    print(f"🎯 Features de fidélité suggérées: {loyalty_features}")
    print(f"🔄 Features alternatives: {alternative_features}")
    
    # Tentative de chargement et analyse des colonnes disponibles
    try:
        # Charger pour inspecter les colonnes
        import pandas as pd
        df_inspect = pd.read_csv(data_path)
        available_cols = list(df_inspect.columns)
        
        print(f"\n📊 COLONNES DISPONIBLES DANS VOS DONNÉES:")
        print("-" * 45)
        for i, col in enumerate(available_cols, 1):
            print(f"   {i:2d}. {col}")
        
        # Identifier automatiquement les features numériques appropriées
        numeric_cols = df_inspect.select_dtypes(include=['number']).columns.tolist()
        
        # Exclure les colonnes ID et target si elles existent
        exclude_patterns = ['id', 'ID', 'index', 'target', 'label', 'fraud', 'loyal', 'churn']
        suggested_features = []
        
        for col in numeric_cols:
            if not any(pattern.lower() in col.lower() for pattern in exclude_patterns):
                suggested_features.append(col)
        
        if len(suggested_features) >= 2:
            print(f"\n✅ FEATURES NUMÉRIQUES DÉTECTÉES AUTOMATIQUEMENT:")
            print("-" * 50)
            for i, feature in enumerate(suggested_features, 1):
                print(f"   {i}. {feature}")
            
            # Utiliser les features détectées
            final_features = suggested_features[:6]  # Prendre les 6 premières max
            
            print(f"\n🎯 FEATURES SÉLECTIONNÉES POUR L'ANALYSE:")
            print(f"   {final_features}")
            
            # Lancer l'analyse avec détection automatique du nombre optimal de clusters
            print(f"\n🚀 LANCEMENT DE L'ANALYSE DE SEGMENTATION...")
            print("-" * 50)
            
            # Test avec différents nombres de clusters
            best_results = None
            best_score = -1
            best_k = 5
            
            print("🔍 Recherche du nombre optimal de clusters...")
            for k in range(3, 8):  # Test de 3 à 7 clusters
                print(f"   Testant {k} clusters...")
                temp_results = quick_segmentation_analysis(
                    data_path=data_path,
                    features_list=final_features,
                    n_clusters=k
                )
                
                if temp_results[0] is not None:
                    evaluation_df = temp_results[2]
                    avg_silhouette = evaluation_df['Silhouette_Score'].mean()
                    
                    if avg_silhouette > best_score:
                        best_score = avg_silhouette
                        best_results = temp_results
                        best_k = k
            
            if best_results is not None:
                print(f"\n🏆 RÉSULTATS OPTIMAUX TROUVÉS!")
                print(f"   Nombre optimal de clusters: {best_k}")
                print(f"   Score Silhouette moyen: {best_score:.4f}")
                
                segmentator, df_with_clusters, evaluation = best_results
                
                # Analyse spécifique pour la fidélité client
                print(f"\n📊 ANALYSE SPÉCIFIQUE FIDÉLITÉ CLIENT")
                print("-" * 40)
                
                # Profil des segments de fidélité
                loyalty_profiles = create_loyalty_segments_profile(df_with_clusters, final_features)
                print(loyalty_profiles)
                
                return best_results
            else:
                print("❌ Impossible de générer une segmentation optimale")
                return None
        else:
            print(f"\n❌ Pas assez de features numériques détectées.")
            print(f"💡 Features numériques trouvées: {numeric_cols}")
            print(f"📝 Vérifiez votre fichier de données ou ajustez manuellement les features")
            return None
            
    except FileNotFoundError:
        print(f"\n❌ FICHIER NON TROUVÉ: {data_path}")
        print(f"📝 VÉRIFICATIONS NÉCESSAIRES:")
        print(f"   1. Le fichier existe-t-il à cet emplacement?")
        print(f"   2. Le chemin est-il correct?")
        print(f"   3. Avez-vous exécuté le preprocessing avant?")
        
        # Proposer de créer des données d'exemple basées sur votre contexte
        print(f"\n🔧 Génération de données d'exemple de fidélité client...")
        create_loyalty_example_data()
        return analyze_customer_loyalty_data()  # Relancer avec les données d'exemple
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        print(f"💡 Vérifiez le format de vos données")
        return None


def create_loyalty_segments_profile(df_with_clusters, features):
    """
    Crée un profil détaillé des segments de fidélité
    """
    profiles_text = []
    
    for cluster_id in sorted(df_with_clusters['Cluster'].unique()):
        if cluster_id == -1:  # Skip outliers
            continue
            
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        size = len(cluster_data)
        percentage = (size / len(df_with_clusters)) * 100
        
        profiles_text.append(f"\n🎯 SEGMENT {cluster_id} ({size} clients - {percentage:.1f}%)")
        profiles_text.append("-" * 45)
        
        for feature in features:
            if feature in cluster_data.columns:
                mean_val = cluster_data[feature].mean()
                std_val = cluster_data[feature].std()
                profiles_text.append(f"   {feature}: {mean_val:.2f} (±{std_val:.2f})")
        
        # Interprétation du segment
        if 'Total_Spent' in features or any('spent' in f.lower() for f in features):
            spending_col = next((f for f in features if 'spent' in f.lower()), features[0])
            avg_spending = cluster_data[spending_col].mean()
            
            if avg_spending > df_with_clusters[spending_col].quantile(0.8):
                interpretation = "💎 CLIENTS PREMIUM - Très haute valeur"
            elif avg_spending > df_with_clusters[spending_col].quantile(0.6):
                interpretation = "⭐ CLIENTS FIDÈLES - Valeur élevée"  
            elif avg_spending > df_with_clusters[spending_col].quantile(0.4):
                interpretation = "🔄 CLIENTS RÉGULIERS - Valeur moyenne"
            else:
                interpretation = "🌱 NOUVEAUX/OCCASIONNELS - Potentiel de croissance"
                
            profiles_text.append(f"   Profil: {interpretation}")
    
    return "\n".join(profiles_text)


def create_loyalty_example_data():
    """
    Crée des données d'exemple basées sur votre contexte de fidélité client
    """
    import numpy as np
    import pandas as pd
    import os
    
    np.random.seed(42)
    n_customers = 3000  # Comme dans votre configuration
    
    # Générer des données réalistes de fidélité client
    data = {
        'customer_id': range(2001, 2001 + n_customers),
        'Age': np.random.normal(35, 12, n_customers).astype(int),
        'Total_Spent': np.random.lognormal(6, 1, n_customers),
        'Purchase_Frequency': np.random.poisson(8, n_customers),
        'Days_Since_Last_Purchase': np.random.exponential(45, n_customers).astype(int),
        'Average_Order_Value': np.random.gamma(2, 50, n_customers),
        'Tenure_Days': np.random.uniform(30, 1095, n_customers).astype(int),  # 1 mois à 3 ans
        'loyalty_score': np.random.uniform(0, 1, n_customers)  # Score de fidélité
    }
    
    df = pd.DataFrame(data)
    
    # Appliquer des contraintes réalistes
    df['Age'] = df['Age'].clip(18, 75)
    df['Total_Spent'] = df['Total_Spent'].clip(50, 10000).round(2)
    df['Days_Since_Last_Purchase'] = df['Days_Since_Last_Purchase'].clip(1, 365)
    df['Average_Order_Value'] = df['Average_Order_Value'].clip(20, 500).round(2)
    
    # Créer le dossier si nécessaire
    os.makedirs("data", exist_ok=True)
    
    # Sauvegarder
    df.to_csv("data/cleaned_customer_loyalty_data.csv", index=False)
    print("✅ Données d'exemple de fidélité client créées: data/cleaned_customer_loyalty_data.csv")
    print(f"📊 {len(df)} clients avec {len(df.columns)-1} features générés")


# Fonction principale adaptée à votre contexte
def main_loyalty_analysis():
    """
    Fonction principale pour l'analyse de fidélité client
    """
    print("🚀 DÉMARRAGE DE L'ANALYSE DE SEGMENTATION CLIENT")
    print("="*60)
    print("Basé sur votre script process.py de machine learning")
    print("Configuration automatique pour données de fidélité client")
    print("="*60)
    
    results = analyze_customer_loyalty_data()
    
    if results and results[0] is not None:
        segmentator, df_with_clusters, evaluation = results
        
        print(f"\n🎉 ANALYSE TERMINÉE AVEC SUCCÈS!")
        print(f"📊 {len(df_with_clusters)} clients segmentés")
        print(f"🏆 Meilleur modèle sélectionné automatiquement")
        print(f"💾 Tous les résultats sauvegardés")
        print(f"📈 Visualisations générées")
        
        print(f"\n💡 PROCHAINES ÉTAPES:")
        print(f"   1. Examinez les visualisations générées")
        print(f"   2. Consultez les fichiers CSV créés") 
        print(f"   3. Utilisez les segments pour vos campagnes marketing")
        print(f"   4. Intégrez dans votre pipeline ML existant")
        
    return results


def demo_data():
    """Génère des données d'exemple pour tester"""
    np.random.seed(42)
    n_customers = 1000
    
    data = {
        'Customer_ID': range(2001, 2001 + n_customers),
        'Age': np.random.normal(35, 10, n_customers).astype(int),
        'Total_Spent': np.random.lognormal(6, 1, n_customers),
        'Purchase_Frequency': np.random.poisson(5, n_customers),
        'Recency': np.random.exponential(30, n_customers).astype(int)
    }
    
    df = pd.DataFrame(data)
    df['Age'] = df['Age'].clip(18, 70)
    df['Total_Spent'] = df['Total_Spent'].clip(50, 5000)
    df['Recency'] = df['Recency'].clip(1, 365)
    
    df.to_csv("customers_dataset.csv", index=False)
    print("✅ Fichier d'exemple créé: customers_dataset.csv")
    
    # Tester avec les données d'exemple
    results = quick_segmentation_analysis(
        data_path="customers_dataset.csv",
        features_list=['Age', 'Total_Spent', 'Purchase_Frequency'],
        n_clusters=5
    )
    
    return results


if __name__ == "__main__":
    main_loyalty_analysis()