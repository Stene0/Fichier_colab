# ============================================================
# API SOLAIRE — Serveur Flask pour votre modèle IA entraîné
# ============================================================
# ÉTAPES :
#   1. Exécutez d'abord le bloc "Export du modèle" dans Google Colab
#   2. Téléchargez le fichier model_solaire.pkl sur votre ordinateur
#   3. Placez model_solaire.pkl dans le même dossier que ce fichier
#   4. Installez les dépendances : pip install flask flask-cors scikit-learn joblib
#   5. Lancez le serveur : python api_solaire.py
#   6. L'API sera accessible sur http://localhost:5000
# ============================================================

# ── BLOC À EXÉCUTER DANS GOOGLE COLAB POUR EXPORTER LE MODÈLE ──
# import joblib
# joblib.dump(model, 'model_solaire.pkl')
# from google.colab import files
# files.download('model_solaire.pkl')
# ────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Autorise les requêtes depuis le fichier HTML local

# Chargement du modèle
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_solaire.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Modèle chargé depuis : {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ ERREUR : Fichier model_solaire.pkl introuvable.")
    print(f"   Placez-le dans : {os.path.dirname(os.path.abspath(__file__))}")
    model = None


@app.route('/health', methods=['GET'])
def health():
    """Endpoint de vérification — ouvrez http://localhost:5000/health dans votre navigateur"""
    if model is None:
        return jsonify({"status": "error", "message": "Modèle non chargé"}), 500
    return jsonify({"status": "ok", "message": "API solaire opérationnelle"})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint principal de prédiction.

    Entrée JSON attendue :
    {
        "consommation_kwh": 8.5,    ← Consommation journalière (kWh/j)
        "puissance_kw": 2.35        ← Puissance instantanée TOTALE (kW)
    }

    Ces deux valeurs correspondent exactement aux colonnes X
    utilisées lors de l'entraînement :
        X[0] = CONSOMMATION EN KWH PAR LA JIRAMA ...
        X[1] = PUISSANCE INSTANTANNE TOTAL

    Sortie JSON :
    {
        "batterie_kwh": 12.5,
        "panneau_wc": 3500,
        "onduleur_kw": 3.0,
        "inputs": { ... }   ← Les valeurs envoyées au modèle (pour vérification)
    }
    """
    if model is None:
        return jsonify({"error": "Modèle non chargé. Vérifiez model_solaire.pkl"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Corps JSON manquant"}), 400

    # Validation des champs requis
    required = ['consommation_kwh', 'puissance_kw']
    for field in required:
        if field not in data:
            return jsonify({"error": f"Champ manquant : {field}"}), 400

    consommation_kwh = float(data['consommation_kwh'])
    puissance_kw     = float(data['puissance_kw'])

    # Construction du vecteur X — même ordre que l'entraînement :
    # X_columns = [
    #   'CONSOMMATION EN KWH PAR LA JIRAMA ...',  → consommation_kwh
    #   'PUISSANCE INSTANTANNE TOTAL'              → puissance_kw (converti en kW)
    # ]
    X = np.array([[consommation_kwh, puissance_kw]])

    # Prédiction : y = [BATTERIE, PANNEAU, ONDULEUR]
    prediction = model.predict(X)[0]

    batterie = float(prediction[0])
    panneau  = float(prediction[1])
    onduleur = float(prediction[2])

    return jsonify({
        "batterie_kwh": round(batterie, 2),
        "panneau_wc":   round(panneau, 0),
        "onduleur_kw":  round(onduleur, 2),
        "conseil": generer_conseil(consommation_kwh, puissance_kw, batterie, panneau, onduleur),
        "inputs": {
            "consommation_kwh": consommation_kwh,
            "puissance_kw": puissance_kw
        }
    })


def generer_conseil(conso, puissance, batterie, panneau, onduleur):
    """Génère un conseil automatique basé sur les valeurs prédites."""
    if puissance > 5:
        return f"Installation de forte puissance ({puissance:.1f} kW). Prévoyez un câblage adapté et un disjoncteur différentiel."
    elif batterie > 20:
        return f"Capacité batterie importante ({batterie:.1f} kWh). Optez pour des batteries lithium LiFePO4 pour une meilleure durée de vie."
    elif conso < 3:
        return "Faible consommation journalière. Un système compact suffit. Vérifiez l'orientation des panneaux (plein nord à Madagascar)."
    else:
        return f"Installation standard. Panneau {panneau:.0f} Wc + batterie {batterie:.1f} kWh + onduleur {onduleur:.1f} kW recommandés."


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  SERVEUR API SOLAIRE")
    print("="*50)
    print(f"  URL principale : http://localhost:5000/predict")
    print(f"  Vérification   : http://localhost:5000/health")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
