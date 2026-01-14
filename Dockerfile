# Utilisation d'une image Python légère
FROM python:3.9-slim

# Définition du répertoire de travail
WORKDIR /app

# Installation des dépendances système nécessaires (si besoin pour numpy/pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie du fichier de dépendances
COPY requirements.txt .

# Installation des bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie de tout le code source dans le conteneur
COPY . .

# Note : Dans une pipeline Jenkins, le fichier 'model_output.pkl' 
# doit idéalement être généré ou récupéré avant le build.
# Si vous voulez forcer l'entraînement au build (non recommandé pour de gros modèles) :
# RUN python train_model.py

# Exposer le port utilisé par FastAPI (défini dans main.py)
EXPOSE 8002

# Commande de lancement de l'application
CMD ["python", "main.py"]
