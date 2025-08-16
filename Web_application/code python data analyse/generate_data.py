import pandas as pd
import numpy as np
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from faker import Faker

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EcommerceDataGenerator:
    """
    Générateur de données synthétiques
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialise le générateur
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.fake = Faker()
        Faker.seed(random_seed)
        
        # Configuration des données de base
        self.setup_base_data()
        
        # Stockage des DataFrames générés
        self.datasets = {}
        
    def setup_base_data(self):
        """Configure les données de base utilisées dans la génération"""
        
        # Configuration produits
        self.product_names = [
            'T-shirt', 'Jeans', 'Sneakers', 'Jacket', 'Hat', 'Socks', 
            'Backpack', 'Belt', 'Dress', 'Boots', 'Watch', 'Sunglasses',
            'Laptop', 'Phone', 'Headphones', 'Tablet', 'Camera', 'Book',
            'Perfume', 'Jewelry'
        ]
        
        self.categories = {
            'T-shirt': 'Clothing', 'Jeans': 'Clothing', 'Dress': 'Clothing',
            'Sneakers': 'Footwear', 'Boots': 'Footwear',
            'Jacket': 'Outerwear',
            'Hat': 'Accessories', 'Socks': 'Accessories', 'Belt': 'Accessories',
            'Watch': 'Accessories', 'Sunglasses': 'Accessories', 'Jewelry': 'Accessories',
            'Backpack': 'Bags',
            'Laptop': 'Electronics', 'Phone': 'Electronics', 'Headphones': 'Electronics',
            'Tablet': 'Electronics', 'Camera': 'Electronics',
            'Book': 'Media', 'Perfume': 'Beauty'
        }
        
        self.brands = [
            'TechCorp', 'StyleBrand', 'ComfortWear', 'UrbanStyle', 'SportMax',
            'ElegantLife', 'ModernTech', 'ClassicBrand', 'TrendyFashion', 'QualityPlus'
        ]
        
        # Configuration clients
        self.customer_names = [
            # Noms masculins
            'Alexander', 'Benjamin', 'Christopher', 'David', 'Ethan', 'Frank', 'George', 
            'Henry', 'Isaac', 'James', 'Kevin', 'Liam', 'Marcus', 'Nathan', 'Oliver', 
            'Patrick', 'Quinn', 'Robert', 'Samuel', 'Thomas', 'Victor', 'William',
            # Noms féminins
            'Alice', 'Bella', 'Charlotte', 'Diana', 'Emma', 'Fiona', 'Grace', 'Hannah',
            'Isabella', 'Julia', 'Katherine', 'Luna', 'Maria', 'Nina', 'Olivia', 'Penny',
            'Rachel', 'Sophia', 'Tina', 'Victoria', 'Wendy', 'Zoe'
        ]
        
        self.female_names = {
            'Alice', 'Bella', 'Charlotte', 'Diana', 'Emma', 'Fiona', 'Grace', 'Hannah',
            'Isabella', 'Julia', 'Katherine', 'Luna', 'Maria', 'Nina', 'Olivia', 'Penny',
            'Rachel', 'Sophia', 'Tina', 'Victoria', 'Wendy', 'Zoe'
        }
        
        self.locations = [
            'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
            'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
            'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis',
            'Seattle', 'Denver', 'Washington DC', 'Boston', 'El Paso', 'Detroit',
            'Nashville', 'Portland', 'Oklahoma City', 'Las Vegas', 'Louisville',
            'Baltimore', 'Milwaukee', 'Albuquerque', 'Tucson', 'Fresno', 'Sacramento',
            'Long Beach', 'Kansas City', 'Mesa', 'Atlanta', 'Colorado Springs', 'Virginia Beach'
        ]
        
        # Configuration marketing
        self.marketing_channels = [
            'Online', 'In-Store', 'Social', 'Email', 'TV', 'Radio', 
            'Billboard', 'YouTube', 'Instagram', 'Facebook'
        ]
        
        self.sales_channels = ['Online', 'In-Store', 'Mobile App', 'Phone']
        
    def generate_products(self, n_products: int = 1000) -> pd.DataFrame:
        """
        Génère des données de produits
        """
        logger.info(f"Génération de {n_products} produits...")
        
        products_data = []
        
        for product_id in range(101, 101 + n_products):
            name = random.choice(self.product_names)
            category = self.categories[name]
            brand = random.choice(self.brands)
            
            # Prix basé sur la catégorie
            price_ranges = {
                'Electronics': (50, 500),
                'Clothing': (15, 80),
                'Footwear': (30, 150),
                'Outerwear': (40, 200),
                'Accessories': (10, 100),
                'Bags': (25, 120),
                'Media': (5, 30),
                'Beauty': (15, 60)
            }
            
            min_price, max_price = price_ranges.get(category, (10, 100))
            price = round(np.random.uniform(min_price, max_price), 2)
            
            # Ajout de variabilité dans les noms
            if random.random() < 0.3:
                name = f"{name} {random.choice(['Pro', 'Lite', 'Plus', 'Max', 'Mini'])}"
            
            products_data.append([
                product_id, name, category, price, brand
            ])
        
        df = pd.DataFrame(products_data, columns=[
            'Product_ID', 'Product_Name', 'Category', 'Price', 'Brand'
        ])
        
        self.datasets['products'] = df
        logger.info(f"✓ {len(df)} produits générés")
        return df
    
    def generate_customers(self, n_customers: int = 1000) -> pd.DataFrame:
        """
        Génère des données de clients
        """
        logger.info(f"Génération de {n_customers} clients...")
        
        customers_data = []
        
        for customer_id in range(2001, 2001 + n_customers):
            name = random.choice(self.customer_names)
            gender = 'Female' if name in self.female_names else 'Male'
            
            # Distribution d'âge plus réaliste
            age = int(np.random.normal(35, 12))
            age = max(18, min(75, age))  # Contraindre entre 18 et 75 ans
            
            location = random.choice(self.locations)
            
            # Date d'inscription plus récente
            join_date = self.fake.date_between(start_date='-2y', end_date='today')
            
            # Montant dépensé corrélé avec l'âge et l'ancienneté
            days_since_join = (datetime.now().date() - join_date).days
            base_spending = np.random.gamma(2, 200)  # Distribution gamma plus réaliste
            age_factor = 1 + (age - 30) * 0.01  # Plus âgé = plus de dépenses
            time_factor = 1 + days_since_join * 0.0005  # Plus ancien = plus de dépenses
            
            total_spent = round(base_spending * age_factor * time_factor, 2)
            total_spent = max(50, min(5000, total_spent))  # Contraindre les valeurs
            
            customers_data.append([
                customer_id, name, age, gender, location, join_date, total_spent
            ])
        
        df = pd.DataFrame(customers_data, columns=[
            'Customer_ID', 'Name', 'Age', 'Gender', 'Location', 'Join_Date', 'Total_Spent'
        ])
        
        self.datasets['customers'] = df
        logger.info(f"✓ {len(df)} clients générés")
        return df
    
    def generate_sales(self, n_sales: int = 2000) -> pd.DataFrame:
        """
        Génère des données de ventes
        """
        logger.info(f"Génération de {n_sales} ventes...")
        
        # Vérifier que les produits et clients existent
        if 'products' not in self.datasets or 'customers' not in self.datasets:
            logger.warning("Génération des produits et clients requis...")
            self.generate_products()
            self.generate_customers()
        
        product_ids = self.datasets['products']['Product_ID'].tolist()
        customer_ids = self.datasets['customers']['Customer_ID'].tolist()
        products_info = self.datasets['products'].set_index('Product_ID').to_dict('index')
        
        sales_data = []
        
        for sale_id in range(1, n_sales + 1):
            product_id = random.choice(product_ids)
            customer_id = random.choice(customer_ids)
            
            # Date de vente plus récente
            date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 400))
            
            # Quantité avec distribution plus réaliste
            quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.25, 0.15, 0.07, 0.03])
            
            # Prix basé sur le produit avec variabilité
            base_price = products_info[product_id]['Price']
            # Appliquer une remise occasionnelle
            discount_factor = 1.0
            if random.random() < 0.2:  # 20% de chance de remise
                discount_factor = random.uniform(0.8, 0.95)
            
            unit_price = round(base_price * discount_factor, 2)
            total_price = round(quantity * unit_price, 2)
            
            channel = random.choice(self.sales_channels)
            
            sales_data.append([
                sale_id, product_id, customer_id, date.strftime('%Y-%m-%d'),
                quantity, total_price, channel
            ])
        
        df = pd.DataFrame(sales_data, columns=[
            'Sale_ID', 'Product_ID', 'Customer_ID', 'Date',
            'Quantity', 'Sale_Price', 'Channel'
        ])
        
        self.datasets['sales'] = df
        logger.info(f"✓ {len(df)} ventes générées")
        return df
    
    def generate_marketing_campaigns(self, n_campaigns: int = 500) -> pd.DataFrame:
        """
        Génère des données de campagnes marketing
        """
        logger.info(f"Génération de {n_campaigns} campagnes marketing...")
        
        campaigns_data = []
        
        for campaign_id in range(1, n_campaigns + 1):
            channel = random.choice(self.marketing_channels)
            
            # Générer dates de campagne cohérentes
            start_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 300))
            duration = random.randint(7, 45)  # Entre 1 semaine et 6 semaines
            end_date = start_date + timedelta(days=duration)
            
            # Budget basé sur le canal
            channel_budgets = {
                'TV': (2000, 10000),
                'Radio': (500, 3000),
                'Billboard': (1000, 5000),
                'Online': (300, 2000),
                'Social': (200, 1500),
                'Email': (100, 800),
                'YouTube': (400, 2500),
                'Instagram': (300, 2000),
                'Facebook': (250, 1800),
                'In-Store': (500, 2000)
            }
            
            min_budget, max_budget = channel_budgets.get(channel, (500, 3000))
            budget = random.randint(min_budget, max_budget)
            
            # Impressions basées sur le budget et le canal
            impressions_per_dollar = {
                'TV': 50, 'Radio': 80, 'Billboard': 30, 'Online': 100,
                'Social': 120, 'Email': 200, 'YouTube': 90, 'Instagram': 110,
                'Facebook': 130, 'In-Store': 40
            }
            
            base_impressions = budget * impressions_per_dollar.get(channel, 80)
            impressions = int(base_impressions * random.uniform(0.8, 1.2))
            
            # Taux de clics réalistes par canal
            ctr_ranges = {
                'TV': (0.001, 0.003),
                'Radio': (0.002, 0.005),
                'Billboard': (0.0005, 0.002),
                'Online': (0.02, 0.05),
                'Social': (0.015, 0.04),
                'Email': (0.03, 0.08),
                'YouTube': (0.01, 0.03),
                'Instagram': (0.02, 0.06),
                'Facebook': (0.015, 0.045),
                'In-Store': (0.05, 0.12)
            }
            
            min_ctr, max_ctr = ctr_ranges.get(channel, (0.01, 0.05))
            ctr = random.uniform(min_ctr, max_ctr)
            clicks = int(impressions * ctr)
            
            # Taux de conversion
            conversion_rate = random.uniform(0.05, 0.25)
            conversions = int(clicks * conversion_rate)
            
            campaigns_data.append([
                campaign_id, channel, start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'), budget, impressions, clicks, conversions
            ])
        
        df = pd.DataFrame(campaigns_data, columns=[
            'Campaign_ID', 'Channel', 'Start_Date', 'End_Date',
            'Budget', 'Impressions', 'Clicks', 'Conversions'
        ])
        
        self.datasets['marketing'] = df
        logger.info(f"✓ {len(df)} campagnes marketing générées")
        return df
    
    def generate_all_datasets(self, 
                            n_products: int = 1500,
                            n_customers: int = 3000, 
                            n_sales: int = 8000,
                            n_campaigns: int = 1200) -> Dict[str, pd.DataFrame]:
        """
        Génère tous les datasets
        """
        logger.info("=== GÉNÉRATION DE TOUS LES DATASETS ===")
        
        # Générer dans l'ordre logique
        self.generate_products(n_products)
        self.generate_customers(n_customers)
        self.generate_sales(n_sales)
        self.generate_marketing_campaigns(n_campaigns)
        
        logger.info("=== GÉNÉRATION TERMINÉE ===")
        return self.datasets
    
    def save_datasets(self, output_dir: str = "data") -> bool:
        """
        Sauvegarde tous les datasets générés en CSV
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            for name, df in self.datasets.items():
                file_path = output_path / f"{name}_dataset.csv"
                df.to_csv(file_path, index=False)
                logger.info(f"✓ {name} sauvegardé: {file_path}")
            
            logger.info(f"Tous les datasets sauvegardés dans: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            return False
    
    def get_dataset_summary(self) -> pd.DataFrame:
        """
        Retourne un résumé des datasets générés
        """
        summary_data = []
        
        for name, df in self.datasets.items():
            summary_data.append([
                name.capitalize(),
                len(df),
                len(df.columns),
                df.memory_usage(deep=True).sum() / 1024**2  # MB
            ])
        
        summary_df = pd.DataFrame(summary_data, columns=[
            'Dataset', 'Rows', 'Columns', 'Memory (MB)'
        ])
        
        return summary_df
    
    def display_sample_data(self, n_rows: int = 5):
        """
        Affiche un échantillon de chaque dataset
        """
        for name, df in self.datasets.items():
            print(f"\n{'='*20} {name.upper()} SAMPLE {'='*20}")
            print(df.head(n_rows))
            print(f"Shape: {df.shape}")


def main():
    """
    Fonction principale pour démontrer l'utilisation du générateur
    """
    # Initialiser le générateur
    generator = EcommerceDataGenerator(random_seed=42)
    
    # Générer tous les datasets
    datasets = generator.generate_all_datasets(
        n_products=3000,
        n_customers=3000,
        n_sales=3000,
        n_campaigns=3000
    )
    
    # Afficher un résumé
    print("\n" + "="*50)
    print("RÉSUMÉ DES DATASETS GÉNÉRÉS")
    print("="*50)
    summary = generator.get_dataset_summary()
    print(summary)
    
    # Afficher des échantillons
    generator.display_sample_data(3)
    
    # Sauvegarder les datasets
    generator.save_datasets("generated_data")
    
    return datasets


if __name__ == "__main__":
    datasets = main()