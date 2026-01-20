# 1. Installer les dépendances Python

sudo apt-get update

sudo apt-get install python3-pip python3-dev

# 2. Installer les bibliothèques nécessaires

pip3 install flask numpy Adafruit_DHT

# 3. Créer le dossier templates pour Flask

mkdir templates

# 4. Sauvegarder le code Python

# (copiez le code principal dans surveillance_dht22.py)

# 5. Sauvegarder le HTML

# (copiez le code HTML dans templates/dashboard_dht22.html)

# 6. Lancer le système

python3 surveillance_dht22.py

```

## 🔌 **Branchement DHT22 sur Raspberry Pi 5**

```

DHT22          Raspberry Pi 5

────────────────────────────

VCC (1)   →   Pin 1 (3.3V)

DATA (2)  →   Pin 7 (GPIO4) 

GND (3)   →   Pin 6 (GND)

Note: Ajouter une résistance pull-up 10kΩ entre VCC et DATA
