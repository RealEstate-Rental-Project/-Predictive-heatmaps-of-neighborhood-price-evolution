# Utilisation d'une image Python légère et stable
FROM python:3.10-slim

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Installation des dépendances système nécessaires pour numpy/pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie du fichier de dépendances
COPY requirements.txt .

# Installation des bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie de tout le code source
# Cela inclut les dossiers model/models/ et model/data/ remplis par Jenkins
COPY . .

# Le port d'écoute est configuré à 8000 dans model/app/main.py
EXPOSE 8000

# Commande de lancement pointant vers le dossier model/app/
CMD ["python", "model/app/main.py"]
