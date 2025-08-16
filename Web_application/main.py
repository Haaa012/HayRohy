from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO
from fpdf import FPDF
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

# --- Initialisation ---
app = FastAPI(title="Classification de Fidélité")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Charger le modèle ML
model_path = os.path.join("models", "model.pkl")
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    print("Attention: model.pkl non trouvé, endpoints prédiction désactivés.")

# Exemple dataframe pour clusters
df = pd.DataFrame({
    'Age': np.random.randint(18,70,100),
    'Gender': np.random.choice(['Male','Female'],100),
    'Total_Spent': np.random.uniform(50,1000,100),
    'Quantity': np.random.randint(1,10,100),
    'Channel': np.random.choice(['Email','Social','In-Store','Online'],100),
    'Category': np.random.choice(['Bags','Outerwear','Accessories','Footwear','Clothing'],100),
    'Cluster': np.random.randint(0,5,100),
    'Avg_Price': np.random.uniform(10,200,100)
})

# --- Pydantic ---
class CustomerData(BaseModel):
    Age: int
    Gender: str
    Total_Spent: float
    Quantity: int
    Channel: str
    Category: str
    Cluster: int
    Avg_Price: float

# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/page.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")

    input_df = pd.DataFrame([data.dict()])

    try:
        prob_loyalty = float(model.predict_proba(input_df)[0][1])
    except:
        prob_loyalty = np.random.rand()
    prob_non_loyalty = 1 - prob_loyalty
    status = "Fidèle" if prob_loyalty >= 0.5 else "Non Fidèle"
    recommendation = "Offrir promotion spéciale" if prob_loyalty < 0.5 else "Fidélité maintenue"

    report_data = {
        "status": status,
        "prob_loyalty": prob_loyalty,
        "prob_non_loyalty": prob_non_loyalty,
        "recommendation": recommendation
    }

    return report_data

@app.post("/download_report")
def download_report(data: dict = Body(...)):
    pdf = FPDF()
    pdf.add_page()
    
    # === LOGO ET EN-TÊTE AMÉLIORÉ ===
    # Définir les chemins possibles pour le logo
    logo_paths = ["hayrohy.png", "static/hayrohy.png", "images/hayrohy.png", "assets/hayrohy.png"]
    logo_found = False
    
    # Chercher le logo dans différents répertoires
    for logo_path in logo_paths:
        if os.path.exists(logo_path):
            try:
                # Ajouter le logo en haut à gauche (taille réduite)
                pdf.image(logo_path, x=15, y=15, w=25)
                logo_found = True
                break
            except Exception as e:
                print(f"Erreur lors du chargement du logo {logo_path}: {e}")
                continue
    
    # Si aucun logo n'est trouvé, créer un placeholder visuel
    if not logo_found:
        # Créer un cercle coloré comme placeholder du logo
        pdf.set_fill_color(35, 47, 123)  # Couleur bleu foncé
        pdf.ellipse(15, 15, 25, 25, 'F')
        
        # Ajouter "HR" en blanc dans le cercle
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(23, 25)
        pdf.cell(9, 5, "HR", 0, 0, 'C')
    
    # Nom de l'entreprise à côté du logo
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(78, 115, 223)  # Couleur bleue
    pdf.set_xy(45, 18)
    pdf.cell(0, 8, "HayRohy", ln=False)
    
    # Titre du rapport sous le nom
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 0, 0)  # Noir pour le titre
    pdf.set_xy(45, 28)
    pdf.cell(0, 8, "Rapport de Fidelite Client", ln=False)
    
    # Ligne de séparation sous l'en-tête avec plus d'espace
    pdf.set_draw_color(78, 115, 223)
    pdf.set_line_width(0.5)
    pdf.line(15, 45, 195, 45)
    
    # Espace supplémentaire avant le contenu
    pdf.ln(25)
    
    # === DONNÉES CLIENT ===
    pdf.set_xy(15, 55)  # Position bien espacée après l'en-tête
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(78, 115, 223)
    pdf.cell(0, 10, "DONNEES CLIENT", ln=True)
    pdf.ln(3)
    
    # En-têtes du tableau avec marges ajustées
    pdf.set_x(15)
    pdf.set_font("Arial", "B", 11)
    pdf.set_fill_color(78, 115, 223)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(85, 10, "Critere", 1, 0, 'C', True)
    pdf.cell(85, 10, "Valeur", 1, 1, 'C', True)
    
    # Données du tableau
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_fill_color(245, 245, 245)
    
    client_data = [
        ("Age", f"{data.get('Age', 'N/A')} ans"),
        ("Genre", data.get('Gender', 'N/A')),
        ("Total Depense", f"${data.get('Total_Spent', 'N/A')}"),
        ("Quantite", f"{data.get('Quantity', 'N/A')} articles"),
        ("Canal", data.get('Channel', 'N/A')),
        ("Categorie", data.get('Category', 'N/A')),
        ("Cluster", f"Cluster {data.get('Cluster', 'N/A')}"),
        ("Prix Moyen", f"${data.get('Avg_Price', 'N/A')}")
    ]
    
    for i, (critere, valeur) in enumerate(client_data):
        pdf.set_x(15)
        fill = i % 2 == 0
        pdf.cell(85, 8, critere, 1, 0, 'L', fill)
        pdf.cell(85, 8, str(valeur), 1, 1, 'L', fill)
    
    pdf.ln(15)
    
    # === RÉSULTATS D'ANALYSE ===
    pdf.set_x(15)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(9, 112, 66)  # Vert pour cette section
    pdf.cell(0, 10, "RESULTATS D'ANALYSE", ln=True)
    pdf.ln(3)
    
    # Récupérer les résultats de prédiction
    try:
        input_df = pd.DataFrame([{
            'Age': data.get('Age', 25),
            'Gender': data.get('Gender', 'Male'),
            'Total_Spent': data.get('Total_Spent', 100),
            'Quantity': data.get('Quantity', 1),
            'Channel': data.get('Channel', 'Online'),
            'Category': data.get('Category', 'Clothing'),
            'Cluster': data.get('Cluster', 0),
            'Avg_Price': data.get('Avg_Price', 50)
        }])
        
        if model is not None:
            prob_loyalty = float(model.predict_proba(input_df)[0][1])
        else:
            prob_loyalty = np.random.rand()
            
        prob_non_loyalty = 1 - prob_loyalty
        status = "Fidele" if prob_loyalty >= 0.5 else "Non Fidele"
        recommendation = "Offrir promotion speciale" if prob_loyalty < 0.5 else "Fidelite maintenue"
        
    except Exception as e:
        prob_loyalty = 0.75
        prob_non_loyalty = 0.25
        status = "Fidele"
        recommendation = "Fidelite maintenue"
    
    # En-têtes du tableau résultats
    pdf.set_x(15)
    pdf.set_font("Arial", "B", 11)
    pdf.set_fill_color(137, 214, 186)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(85, 10, "Metrique", 1, 0, 'C', True)
    pdf.cell(85, 10, "Resultat", 1, 1, 'C', True)
    
    # Données du tableau résultats
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)
    
    results_data = [
        ("Statut", status),
        ("Probabilite Fidelite", f"{prob_loyalty * 100:.2f}%"),
        ("Probabilite Non-Fidelite", f"{prob_non_loyalty * 100:.2f}%"),
        ("Action Recommandee", recommendation)
    ]
    
    for i, (metrique, resultat) in enumerate(results_data):
        pdf.set_x(15)
        fill = i % 2 == 0
        pdf.cell(85, 8, metrique, 1, 0, 'L', fill)
        
        # Colorier le statut selon le résultat
        if metrique == "Statut":
            if "Fidele" in status:
                pdf.set_text_color(0, 128, 0)  # Vert pour fidèle
            else:
                pdf.set_text_color(255, 0, 0)  # Rouge pour non fidèle
        
        pdf.cell(85, 8, str(resultat), 1, 1, 'L', fill)
        pdf.set_text_color(0, 0, 0)
    
    pdf.ln(20)
    
    # === VISUALISATIONS PAR CLUSTER ===
    pdf.set_x(15)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(78, 115, 223)
    pdf.cell(0, 10, "VISUALISATIONS PAR CLUSTER", ln=True)
    pdf.ln(8)
    
    try:
        # Vérifier s'il y a assez d'espace pour les graphiques
        current_y = pdf.get_y()
        space_needed = 160  # Espace nécessaire pour 2 graphiques
        
        if current_y > 280 - space_needed:  # Si pas assez d'espace
            pdf.add_page()
            pdf.ln(10)
            current_y = pdf.get_y()
        
        # Générer le graphique des dépenses avec marges optimisées
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        df.groupby('Cluster')['Total_Spent'].mean().plot(kind='bar', ax=ax1, color='#4e73df')
        ax1.set_ylabel("Depense moyenne ($)", fontsize=10)
        ax1.set_xlabel("Cluster", fontsize=10)
        ax1.set_title("Depenses Moyennes par Cluster", fontsize=12, pad=15)
        ax1.tick_params(axis='x', rotation=0, labelsize=9)
        ax1.tick_params(axis='y', labelsize=9)
        ax1.grid(True, alpha=0.3)
        
        spent_chart_path = "temp_spent_chart.png"
        plt.tight_layout(pad=2.0)
        plt.savefig(spent_chart_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig1)
        
        # Ajouter le premier graphique
        pdf.image(spent_chart_path, x=20, y=current_y, w=150)
        pdf.ln(80)  # Espace suffisant pour le graphique
        
        # Vérifier l'espace pour le deuxième graphique
        current_y = pdf.get_y()
        if current_y > 200:  # Si trop bas, nouvelle page
            pdf.add_page()
            pdf.ln(10)
        
        # Générer le graphique des âges avec marges optimisées
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        df.groupby('Cluster')['Age'].mean().plot(kind='bar', ax=ax2, color='#1cc88a')
        ax2.set_ylabel("Age moyen (ans)", fontsize=10)
        ax2.set_xlabel("Cluster", fontsize=10)
        ax2.set_title("Age Moyen par Cluster", fontsize=12, pad=15)
        ax2.tick_params(axis='x', rotation=0, labelsize=9)
        ax2.tick_params(axis='y', labelsize=9)
        ax2.grid(True, alpha=0.3)
        
        age_chart_path = "temp_age_chart.png"
        plt.tight_layout(pad=2.0)
        plt.savefig(age_chart_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig2)
        
        # Ajouter le deuxième graphique
        current_y = pdf.get_y()
        pdf.image(age_chart_path, x=20, y=current_y, w=150)
        pdf.ln(80)
        
        # Nettoyer les fichiers temporaires
        for temp_file in [spent_chart_path, age_chart_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
    except Exception as e:
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f"Erreur lors de la generation des graphiques: {str(e)}", ln=True)
        pdf.ln(10)
    
    # Footer avec informations de l'entreprise
    pdf.ln(15)
    
    # Ligne de séparation avant le footer
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # Informations footer
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(78, 115, 223)
    pdf.cell(0, 5, "HayRohy - Solution de Classification de Fidelite", ln=True, align="C")
    
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(128, 128, 128)
    from datetime import datetime
    pdf.cell(0, 5, f"Rapport genere le {datetime.now().strftime('%d/%m/%Y a %H:%M')}", ln=True, align="C")

    pdf_path = "PDF_rapport/rapport_client_fidelite.pdf" # fichier raport enregistrer 
    pdf.output(pdf_path)

    return FileResponse(pdf_path, media_type='application/pdf', filename="rapport_fidelite.pdf")

# --- Cluster summary ---
def get_cluster_summary(df: pd.DataFrame):
    summary = []
    for cluster in sorted(df['Cluster'].unique()):
        cdata = df[df['Cluster']==cluster]
        summary.append({
            'Cluster': int(cluster),
            'Nb_clients': int(cdata.shape[0]),
            'Age_moyen': float(round(cdata['Age'].mean(),2)),
            'Depense_moyenne': float(round(cdata['Total_Spent'].mean(),2)),
            'Depense_max': float(round(cdata['Total_Spent'].max(),2))
        })
    return summary

@app.get("/cluster_summary")
async def cluster_summary():
    return get_cluster_summary(df)

@app.get("/recommendations")
async def recommendations():
    recs = []
    for cluster in sorted(df['Cluster'].unique()):
        recs.append({
            "cluster": int(cluster),
            "recommendation": f"Recommandation pour cluster {cluster}: offrir des promotions ciblées."
        })
    return recs

# --- Charts ---
@app.get("/charts/spent.png")
async def spent_chart():
    fig, ax = plt.subplots()
    df.groupby('Cluster')['Total_Spent'].mean().plot(kind='bar', ax=ax)
    ax.set_ylabel("Dépense moyenne")
    ax.set_xlabel("Cluster")
    ax.set_title("Dépense moyenne par cluster")
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png')
    plt.close(fig)
    img_bytes.seek(0)
    return StreamingResponse(img_bytes, media_type="image/png")

@app.get("/charts/age.png")
async def age_chart():
    fig, ax = plt.subplots()
    df.groupby('Cluster')['Age'].mean().plot(kind='bar', ax=ax)
    ax.set_ylabel("Âge moyen")
    ax.set_xlabel("Cluster")
    ax.set_title("Âge moyen par cluster")
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png')
    plt.close(fig)
    img_bytes.seek(0)
    return StreamingResponse(img_bytes, media_type="image/png")