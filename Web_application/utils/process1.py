import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CustomerLoyaltyPredictor:
    """
    Classe pour la prédiction de fidélité client utilisant Random Forest
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipeline = None
        self.is_trained = False
        
    def load_data(self, data_path):
        """
        Charge les données depuis un fichier CSV
        
        Args:
            data_path (str): Chemin vers le fichier de données
            
        Returns:
            pd.DataFrame: DataFrame avec les données chargées
        """
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Données chargées avec succès. Taille: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"Fichier non trouvé: {data_path}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            raise
            
    def prepare_data(self, df):
        """
        Prépare les données en séparant les features et la variable cible
        """
        X = df.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
        y = df.iloc[:, -1]   # Dernière colonne (variable cible)
        
        logger.info(f"Features préparées: {X.shape[1]} variables")
        logger.info(f"Distribution de la variable cible:\n{y.value_counts()}")
        
        return X, y
        
    def build_pipeline(self):
        """
        Construit le pipeline de preprocessing et d'entraînement
        """
        # Sélecteurs de colonnes
        numeric_features = make_column_selector(dtype_include=np.number)
        categorical_features = make_column_selector(dtype_exclude=np.number)
        
        # Preprocessing pour variables numériques
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Preprocessing pour variables catégorielles
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])
        
        # Combinaison des transformations
        preprocessor = ColumnTransformer([
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ])
        
        # Pipeline complet
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ))
        ])
        
        return pipeline
        
    def train_model(self, X, y, test_size=0.2):
        """
        Entraîne le modèle et évalue ses performances
        """
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Construction et entraînement du pipeline
        self.pipeline = self.build_pipeline()
        logger.info("Début de l'entraînement du modèle...")
        
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        # Prédictions
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Évaluation des performances
        results = self.evaluate_model(y_test, y_pred, y_prob)
        results['X_test'] = X_test
        results['y_test'] = y_test
        results['y_pred'] = y_pred
        results['y_prob'] = y_prob
        
        # Validation croisée
        cv_scores = cross_val_score(
            self.pipeline, X_train, y_train, cv=5, scoring='accuracy'
        )
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
        
        logger.info("Entraînement terminé avec succès!")
        return results
        
    def evaluate_model(self, y_true, y_pred, y_prob):
        """
        Évalue les performances du modèle
        """
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_prob)
        
        results = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        return results
        
    def print_results(self, results):
        """
        Affiche les résultats de l'évaluation
        """
        print("=" * 50)
        print("RÉSULTATS DE L'ÉVALUATION")
        print("=" * 50)
        print(f"Précision (Accuracy): {results['accuracy']:.4f}")
        print(f"ROC AUC Score: {results['roc_auc']:.4f}")
        print(f"Validation croisée: {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})")
        print("\nRapport de classification:")
        print(results['classification_report'])
        print("\nMatrice de confusion:")
        print(results['confusion_matrix'])
        
        # Exemples de probabilités
        if 'y_prob' in results:
            print(f"\nExemples de probabilités de fidélité:")
            print(results['y_prob'][:10])
            
    def save_model(self, model_path):
        """
        Sauvegarde le modèle entraîné
        """
        if not self.is_trained:
            logger.error("Le modèle n'a pas été entraîné!")
            return False
            
        try:
            # Créer le répertoire si il n'existe pas
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(self.pipeline, model_path)
            logger.info(f"Modèle sauvegardé: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            return False
            
    def get_feature_importance(self, feature_names=None):
        """
        Récupère l'importance des features du Random Forest
        """
        if not self.is_trained:
            logger.error("Le modèle n'a pas été entraîné!")
            return None
            
        # Récupérer le classificateur du pipeline
        classifier = self.pipeline.named_steps['classifier']
        importances = classifier.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
            
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df


def main():
    """
    Fonction principale pour exécuter le pipeline complet
    """
    # Configuration des chemins
    current_dir = Path.cwd()
    data_path = current_dir / "/home/fa/Documents/M1 AI/SMD/SMD IA/HayRohy_v2/Web_application/data" / "cleaned_data.csv"
    model_path = current_dir / "models" / "model.pkl"
    
    # Initialisation du prédicteur
    predictor = CustomerLoyaltyPredictor(random_state=42)
    
    try:
        # Chargement et préparation des données
        df = predictor.load_data(data_path)
        X, y = predictor.prepare_data(df)
        
        # Entraînement du modèle
        results = predictor.train_model(X, y)
        
        # Affichage des résultats
        predictor.print_results(results)
        
        # Sauvegarde du modèle
        predictor.save_model(model_path)
        
        # Importance des features (optionnel)
        importance_df = predictor.get_feature_importance()
        if importance_df is not None:
            print("\nTop 10 des features les plus importantes:")
            print(importance_df.head(10))
            
    except Exception as e:
        logger.error(f"Erreur dans le pipeline principal: {e}")
        raise


if __name__ == "__main__":
    main()