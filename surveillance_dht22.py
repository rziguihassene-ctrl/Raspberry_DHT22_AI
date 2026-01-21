"""
Système de Surveillance Température et Humidité DHT22
Version: 2.1 - Raspberry Pi 4 Compatible
Utilise adafruit-circuitpython-dht (moderne et maintenu)
"""

import sqlite3
import time
import json
from datetime import datetime, timedelta
from threading import Thread, Lock
from flask import Flask, render_template, jsonify
import numpy as np
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import statistics

# Import pour DHT22 sur Raspberry Pi 4
try:
    import board
    import adafruit_dht
    DHT_DISPONIBLE = True
    print("✅ Bibliothèque adafruit-circuitpython-dht chargée")
except ImportError:
    print("⚠️  adafruit-circuitpython-dht non installé - Mode simulation activé")
    DHT_DISPONIBLE = False


# ============================================================================
# MODÈLE DE DONNÉES
# ============================================================================

@dataclass
class Mesure:
    """Représente une mesure température/humidité"""
    timestamp: str
    temperature: float
    humidite: float
    point_rosee: float
    indice_chaleur: float
    humidite_absolue: float


@dataclass
class Anomalie:
    """Représente une anomalie détectée"""
    timestamp: str
    niveau: str  # normal, avertissement, critique
    type_anomalie: str
    parametre: str
    valeur: float
    ecart_type: float
    message: str
    confiance: float


# ============================================================================
# GESTIONNAIRE DE BASE DE DONNÉES
# ============================================================================

class DatabaseManager:
    """Gestion de la base de données SQLite avec création automatique"""
    
    def __init__(self, db_path: str = "dht22_surveillance.db"):
        self.db_path = db_path
        self.lock = Lock()
        self._creer_tables()
        print(f"✅ Base de données initialisée: {db_path}")
    
    def _creer_tables(self):
        """Crée automatiquement toutes les tables nécessaires"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Table des mesures
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mesures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    temperature REAL,
                    humidite REAL,
                    point_rosee REAL,
                    indice_chaleur REAL,
                    humidite_absolue REAL
                )
            """)
            
            # Table des anomalies
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    niveau TEXT,
                    type_anomalie TEXT,
                    parametre TEXT,
                    valeur REAL,
                    ecart_type REAL,
                    message TEXT,
                    confiance REAL
                )
            """)
            
            # Table des statistiques adaptatives
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS statistiques_adaptatives (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parametre TEXT NOT NULL,
                    moyenne REAL,
                    ecart_type REAL,
                    min_valeur REAL,
                    max_valeur REAL,
                    nb_echantillons INTEGER,
                    derniere_maj TEXT,
                    UNIQUE(parametre)
                )
            """)
            
            # Index pour performances
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mesures_timestamp 
                ON mesures(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp 
                ON anomalies(timestamp)
            """)
            
            conn.commit()
    
    def inserer_mesure(self, mesure: Mesure) -> int:
        """Insère une mesure dans la base de données"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO mesures 
                    (timestamp, temperature, humidite, point_rosee, indice_chaleur, humidite_absolue)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    mesure.timestamp,
                    mesure.temperature,
                    mesure.humidite,
                    mesure.point_rosee,
                    mesure.indice_chaleur,
                    mesure.humidite_absolue
                ))
                conn.commit()
                return cursor.lastrowid
    
    def inserer_anomalie(self, anomalie: Anomalie):
        """Insère une anomalie détectée"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO anomalies 
                    (timestamp, niveau, type_anomalie, parametre, valeur, ecart_type, message, confiance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    anomalie.timestamp,
                    anomalie.niveau,
                    anomalie.type_anomalie,
                    anomalie.parametre,
                    anomalie.valeur,
                    anomalie.ecart_type,
                    anomalie.message,
                    anomalie.confiance
                ))
                conn.commit()
    
    def obtenir_mesures_recentes(self, limite: int = 100) -> List[Dict]:
        """Récupère les mesures les plus récentes"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM mesures 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limite,))
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def obtenir_statistiques(self, heures: int = 24) -> Dict:
        """Calcule les statistiques sur une période"""
        temps_limite = (datetime.now() - timedelta(hours=heures)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as nb_mesures,
                    AVG(temperature) as temp_moy,
                    MIN(temperature) as temp_min,
                    MAX(temperature) as temp_max,
                    AVG(humidite) as hum_moy,
                    MIN(humidite) as hum_min,
                    MAX(humidite) as hum_max,
                    AVG(point_rosee) as rosee_moy
                FROM mesures 
                WHERE timestamp > ?
            """, (temps_limite,))
            
            columns = [description[0] for description in cursor.description]
            result = cursor.fetchone()
            return dict(zip(columns, result)) if result else {}


# ============================================================================
# DÉTECTEUR D'ANOMALIES ADAPTATIF
# ============================================================================

class DetecteurAnomaliesAdaptatif:
    """Détecteur d'anomalies avec apprentissage automatique adaptatif"""
    
    def __init__(self, db: DatabaseManager, taille_fenetre: int = 100):
        self.db = db
        self.taille_fenetre = taille_fenetre
        
        # Fenêtres glissantes pour chaque paramètre
        self.fenetres = {
            'temperature': deque(maxlen=taille_fenetre),
            'humidite': deque(maxlen=taille_fenetre),
            'point_rosee': deque(maxlen=taille_fenetre),
            'indice_chaleur': deque(maxlen=taille_fenetre)
        }
        
        # Statistiques adaptatives
        self.stats = {param: {'moyenne': None, 'ecart_type': None, 'nb_points': 0} 
                     for param in self.fenetres.keys()}
        
        # Phase d'apprentissage initial
        self.phase_apprentissage = True
        self.nb_mesures_apprentissage = 50
        
        # Seuils adaptatifs
        self.seuils = {
            'avertissement': 2.5,
            'critique': 3.5
        }
        
        # Historique des anomalies
        self.historique_anomalies = deque(maxlen=1000)
        
        print("✅ Détecteur d'anomalies adaptatif initialisé")
    
    def ajouter_mesure(self, mesure: Mesure):
        """Ajoute une mesure et met à jour les statistiques"""
        self.fenetres['temperature'].append(mesure.temperature)
        self.fenetres['humidite'].append(mesure.humidite)
        self.fenetres['point_rosee'].append(mesure.point_rosee)
        self.fenetres['indice_chaleur'].append(mesure.indice_chaleur)
        
        # Mettre à jour les statistiques
        for param, valeurs in self.fenetres.items():
            if len(valeurs) >= 10:
                self.stats[param]['moyenne'] = statistics.mean(valeurs)
                self.stats[param]['ecart_type'] = statistics.stdev(valeurs) if len(valeurs) > 1 else 0
                self.stats[param]['nb_points'] = len(valeurs)
        
        # Vérifier si phase d'apprentissage terminée
        if self.phase_apprentissage and len(self.fenetres['temperature']) >= self.nb_mesures_apprentissage:
            self.phase_apprentissage = False
            print(f"✅ Phase d'apprentissage terminée - Détection activée")
    
    def detecter_anomalies(self, mesure: Mesure) -> List[Anomalie]:
        """Détecte les anomalies dans une mesure"""
        anomalies = []
        
        if self.phase_apprentissage:
            return anomalies
        
        timestamp = mesure.timestamp
        
        # Analyser chaque paramètre
        parametres = {
            'temperature': mesure.temperature,
            'humidite': mesure.humidite,
            'point_rosee': mesure.point_rosee,
            'indice_chaleur': mesure.indice_chaleur
        }
        
        for param, valeur in parametres.items():
            stat = self.stats[param]
            
            if stat['moyenne'] is None or stat['ecart_type'] is None:
                continue
            
            # Calculer l'écart en nombre d'écarts-types
            if stat['ecart_type'] > 0:
                z_score = abs(valeur - stat['moyenne']) / stat['ecart_type']
            else:
                z_score = 0
            
            # Détecter anomalie
            niveau = None
            type_anomalie = None
            confiance = 0
            
            if z_score >= self.seuils['critique']:
                niveau = 'critique'
                type_anomalie = 'deviation_extreme'
                confiance = min(0.95, 0.7 + (z_score / 10))
            elif z_score >= self.seuils['avertissement']:
                niveau = 'avertissement'
                type_anomalie = 'deviation_moderee'
                confiance = min(0.85, 0.6 + (z_score / 10))
            
            if niveau:
                message = (f"{param.upper()}: {valeur:.2f} "
                          f"(écart: {z_score:.1f}σ de la moyenne {stat['moyenne']:.2f})")
                
                anomalie = Anomalie(
                    timestamp=timestamp,
                    niveau=niveau,
                    type_anomalie=type_anomalie,
                    parametre=param,
                    valeur=valeur,
                    ecart_type=z_score,
                    message=message,
                    confiance=confiance
                )
                
                anomalies.append(anomalie)
                self.historique_anomalies.append(anomalie)
        
        # Détections spécifiques au climat
        anomalies.extend(self._detecter_conditions_extremes(mesure))
        
        return anomalies
    
    def _detecter_conditions_extremes(self, mesure: Mesure) -> List[Anomalie]:
        """Détecte des conditions climatiques extrêmes"""
        anomalies = []
        
        # Température critique
        if mesure.temperature > 40:
            anomalies.append(Anomalie(
                timestamp=mesure.timestamp,
                niveau='critique',
                type_anomalie='temperature_extreme',
                parametre='temperature',
                valeur=mesure.temperature,
                ecart_type=0,
                message=f"⚠️ TEMPÉRATURE CRITIQUE: {mesure.temperature:.1f}°C - Risque de surchauffe!",
                confiance=0.95
            ))
        elif mesure.temperature < 0:
            anomalies.append(Anomalie(
                timestamp=mesure.timestamp,
                niveau='critique',
                type_anomalie='temperature_extreme',
                parametre='temperature',
                valeur=mesure.temperature,
                ecart_type=0,
                message=f"❄️ TEMPÉRATURE CRITIQUE: {mesure.temperature:.1f}°C - Risque de gel!",
                confiance=0.95
            ))
        
        # Humidité critique
        if mesure.humidite > 90:
            anomalies.append(Anomalie(
                timestamp=mesure.timestamp,
                niveau='avertissement',
                type_anomalie='humidite_elevee',
                parametre='humidite',
                valeur=mesure.humidite,
                ecart_type=0,
                message=f"💧 HUMIDITÉ ÉLEVÉE: {mesure.humidite:.1f}% - Risque de condensation!",
                confiance=0.90
            ))
        elif mesure.humidite < 20:
            anomalies.append(Anomalie(
                timestamp=mesure.timestamp,
                niveau='avertissement',
                type_anomalie='humidite_faible',
                parametre='humidite',
                valeur=mesure.humidite,
                ecart_type=0,
                message=f"🏜️ HUMIDITÉ FAIBLE: {mesure.humidite:.1f}% - Air très sec!",
                confiance=0.90
            ))
        
        return anomalies
    
    def obtenir_etat(self) -> Dict:
        """Retourne l'état du détecteur"""
        return {
            'phase_apprentissage': self.phase_apprentissage,
            'nb_points_collectes': len(self.fenetres['temperature']),
            'statistiques': self.stats,
            'seuils': self.seuils,
            'nb_anomalies_historique': len(self.historique_anomalies)
        }


# ============================================================================
# CAPTEUR DHT22 - VERSION RASPBERRY PI 4
# ============================================================================

class DHT22Sensor:
    """Interface pour le capteur DHT22 sur Raspberry Pi 4"""
    
    def __init__(self, gpio_pin: int = 4, mode_simulation: bool = not DHT_DISPONIBLE):
        self.gpio_pin = gpio_pin
        self.mode_simulation = mode_simulation
        self.dht_device = None
        
        # Variables de simulation
        self.temp_base = 22.0
        self.hum_base = 50.0
        self.compteur = 0
        
        if self.mode_simulation:
            print(f"⚠️  Mode SIMULATION activé (GPIO {gpio_pin})")
        else:
            # Initialiser le capteur DHT22 réel
            try:
                # Correspondance GPIO board pour Pi 4
                gpio_map = {
                    4: board.D4,
                    17: board.D17,
                    27: board.D27,
                    22: board.D22,
                    # Ajoutez d'autres si nécessaire
                }
                
                gpio_board = gpio_map.get(gpio_pin, board.D4)
                self.dht_device = adafruit_dht.DHT22(gpio_board, use_pulseio=False)
                print(f"✅ Capteur DHT22 initialisé sur GPIO {gpio_pin} (Raspberry Pi 4)")
                
            except Exception as e:
                print(f"❌ Erreur initialisation DHT22: {e}")
                print("⚠️  Passage en mode simulation")
                self.mode_simulation = True
                self.dht_device = None
    
    def _calculer_point_rosee(self, temp: float, hum: float) -> float:
        """Calcule le point de rosée (formule Magnus)"""
        a = 17.27
        b = 237.7
        alpha = ((a * temp) / (b + temp)) + np.log(hum/100.0)
        return (b * alpha) / (a - alpha)
    
    def _calculer_indice_chaleur(self, temp: float, hum: float) -> float:
        """Calcule l'indice de chaleur (Heat Index)"""
        if temp < 27:
            return temp
        
        # Formule Rothfusz
        c1, c2, c3 = -8.78469475556, 1.61139411, 2.33854883889
        c4, c5, c6 = -0.14611605, -0.012308094, -0.0164248277778
        c7, c8, c9 = 0.002211732, 0.00072546, -0.000003582
        
        T = temp
        R = hum
        
        HI = (c1 + c2*T + c3*R + c4*T*R + c5*T*T + c6*R*R + 
              c7*T*T*R + c8*T*R*R + c9*T*T*R*R)
        
        return HI
    
    def _calculer_humidite_absolue(self, temp: float, hum: float) -> float:
        """Calcule l'humidité absolue (g/m³)"""
        a = 17.27
        b = 237.7
        
        # Pression de vapeur saturante
        es = 6.112 * np.exp((a * temp) / (b + temp))
        
        # Pression de vapeur réelle
        e = (hum / 100) * es
        
        # Humidité absolue
        return (e * 2.1674) / (temp + 273.15)
    
    def lire_mesure(self) -> Optional[Mesure]:
        """Lit une mesure du capteur DHT22"""
        try:
            if self.mode_simulation:
                # Mode simulation
                self.compteur += 1
                
                # Variations normales
                temp = self.temp_base + np.random.normal(0, 1.5)
                hum = self.hum_base + np.random.normal(0, 5)
                
                # Anomalies périodiques
                if self.compteur % 200 == 0:
                    temp += np.random.choice([8, -8])
                if self.compteur % 300 == 0:
                    hum += np.random.choice([25, -25])
                
                # Contraintes physiques
                temp = max(-20, min(60, temp))
                hum = max(0, min(100, hum))
                
            else:
                # Lecture réelle du capteur DHT22
                try:
                    temp = self.dht_device.temperature
                    hum = self.dht_device.humidity
                    
                    if temp is None or hum is None:
                        return None
                        
                except RuntimeError as e:
                    # Les erreurs de lecture temporaires sont normales avec DHT22
                    # On les ignore et on réessaiera au prochain cycle
                    return None
            
            # Calculs dérivés
            point_rosee = self._calculer_point_rosee(temp, hum)
            indice_chaleur = self._calculer_indice_chaleur(temp, hum)
            hum_absolue = self._calculer_humidite_absolue(temp, hum)
            
            return Mesure(
                timestamp=datetime.now().isoformat(),
                temperature=round(temp, 2),
                humidite=round(hum, 2),
                point_rosee=round(point_rosee, 2),
                indice_chaleur=round(indice_chaleur, 2),
                humidite_absolue=round(hum_absolue, 2)
            )
            
        except Exception as e:
            print(f"❌ Erreur lecture capteur: {e}")
            return None
    
    def __del__(self):
        """Nettoyage lors de la destruction de l'objet"""
        if self.dht_device is not None:
            try:
                self.dht_device.exit()
            except:
                pass


# ============================================================================
# SYSTÈME PRINCIPAL
# ============================================================================

class SystemeSurveillanceClimat:
    """Système principal de surveillance température/humidité"""
    
    def __init__(self, gpio_pin: int = 4, db_path: str = "dht22_surveillance.db"):
        print("🌡️  Initialisation du Système de Surveillance DHT22")
        print("=" * 70)
        
        # Composants
        self.db = DatabaseManager(db_path)
        self.capteur = DHT22Sensor(gpio_pin=gpio_pin)
        self.detecteur = DetecteurAnomaliesAdaptatif(self.db)
        
        # Configuration
        self.intervalle_mesure = 3  # secondes (DHT22 max 0.5Hz = 2s minimum)
        self.running = False
        
        # Dernières données pour l'interface web
        self.dernieres_mesures = deque(maxlen=100)
        self.dernieres_anomalies = deque(maxlen=50)
        self.lock = Lock()
        
        print("=" * 70)
        print("✅ Système initialisé avec succès\n")
    
    def cycle_mesure(self):
        """Exécute un cycle de mesure complet"""
        try:
            # 1. Lire le capteur
            mesure = self.capteur.lire_mesure()
            
            if mesure is None:
                return None, []
            
            # 2. Enregistrer dans la BD
            self.db.inserer_mesure(mesure)
            
            # 3. Ajouter au détecteur
            self.detecteur.ajouter_mesure(mesure)
            
            # 4. Détecter anomalies
            anomalies = self.detecteur.detecter_anomalies(mesure)
            
            # 5. Enregistrer les anomalies
            for anomalie in anomalies:
                self.db.inserer_anomalie(anomalie)
                with self.lock:
                    self.dernieres_anomalies.append(asdict(anomalie))
            
            # 6. Mettre à jour les dernières mesures
            with self.lock:
                self.dernieres_mesures.append(asdict(mesure))
            
            # 7. Afficher l'état
            self._afficher_etat(mesure, anomalies)
            
            return mesure, anomalies
            
        except Exception as e:
            print(f"❌ Erreur cycle de mesure: {e}")
            return None, []
    
    def _afficher_etat(self, mesure: Mesure, anomalies: List[Anomalie]):
        """Affiche l'état actuel en console"""
        heure = datetime.now().strftime('%H:%M:%S')
        
        # Emoji selon anomalies
        if any(a.niveau == 'critique' for a in anomalies):
            emoji = "🚨"
            niveau = "CRITIQUE"
        elif any(a.niveau == 'avertissement' for a in anomalies):
            emoji = "⚠️ "
            niveau = "AVERTISSEMENT"
        else:
            emoji = "✅"
            niveau = "NORMAL"
        
        print(f"{emoji} [{heure}] {niveau} | "
              f"T={mesure.temperature:.1f}°C | "
              f"H={mesure.humidite:.1f}% | "
              f"Rosée={mesure.point_rosee:.1f}°C | "
              f"IC={mesure.indice_chaleur:.1f}°C")
        
        # Afficher anomalies
        for anomalie in anomalies:
            print(f"   └─ {anomalie.message}")
    
    def boucle_surveillance(self):
        """Boucle principale de surveillance"""
        print("🔍 Démarrage de la surveillance...")
        print("Appuyez sur Ctrl+C pour arrêter\n")
        
        self.running = True
        
        try:
            while self.running:
                self.cycle_mesure()
                time.sleep(self.intervalle_mesure)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Arrêt demandé...")
        finally:
            self.arreter()
    
    def arreter(self):
        """Arrête proprement le système"""
        self.running = False
        print("✅ Système arrêté proprement")
    
    def obtenir_donnees_dashboard(self) -> Dict:
        """Retourne les données pour le dashboard web"""
        with self.lock:
            return {
                'mesures_recentes': list(self.dernieres_mesures),
                'anomalies_recentes': list(self.dernieres_anomalies),
                'statistiques': self.db.obtenir_statistiques(heures=24),
                'etat_detecteur': self.detecteur.obtenir_etat()
            }


# ============================================================================
# SERVEUR WEB
# ============================================================================

app = Flask(__name__)
systeme = None

@app.route('/')
def index():
    return render_template('dashboard_dht22.html')

@app.route('/api/donnees')
def api_donnees():
    if systeme:
        return jsonify(systeme.obtenir_donnees_dashboard())
    return jsonify({'error': 'Système non initialisé'}), 503

@app.route('/api/statistiques')
def api_statistiques():
    if systeme:
        return jsonify({
            'stats_24h': systeme.db.obtenir_statistiques(heures=24),
            'stats_7j': systeme.db.obtenir_statistiques(heures=24*7)
        })
    return jsonify({'error': 'Système non initialisé'}), 503


def demarrer_serveur_web(port=5000):
    print(f"\n🌐 Serveur web démarré sur http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Configuration
    GPIO_PIN = 4  # GPIO4 par défaut (pin 7)
    
    # Créer le système
    systeme = SystemeSurveillanceClimat(gpio_pin=GPIO_PIN)
    
    # Démarrer le serveur web dans un thread séparé
    thread_web = Thread(target=demarrer_serveur_web, daemon=True)
    thread_web.start()
    
    # Laisser le temps au serveur de démarrer
    time.sleep(2)
    
    # Lancer la surveillance
    systeme.boucle_surveillance()
