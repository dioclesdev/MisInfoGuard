#!/usr/bin/env python3
"""
ISOT Dataset Download Script
Lädt automatisch das ISOT Fake News Dataset herunter
"""

import requests
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_isot_dataset():
    """
    Lädt das ISOT Fake News Dataset automatisch herunter
    """
    
    # Erstelle datasets Ordner
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    # ISOT Dataset URLs (diese können sich ändern)
    urls = {
        "Fake.csv": "https://raw.githubusercontent.com/rexshell/ISOT-Fake-News-Dataset/master/Fake.csv",
        "True.csv": "https://raw.githubusercontent.com/rexshell/ISOT-Fake-News-Dataset/master/True.csv"
    }
    
    # Alternative URLs falls die ersten nicht funktionieren
    alt_urls = {
        "Fake.csv": "https://huggingface.co/datasets/GonzaloA/fake_news/resolve/main/Fake.csv",
        "True.csv": "https://huggingface.co/datasets/GonzaloA/fake_news/resolve/main/True.csv"
    }
    
    success_count = 0
    
    for filename, url in urls.items():
        filepath = datasets_dir / filename
        
        if filepath.exists():
            logger.info(f"✅ {filename} bereits vorhanden")
            success_count += 1
            continue
        
        try:
            logger.info(f"📥 Lade {filename} von {url}...")
            
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Speichere Datei
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Validiere CSV
            df = pd.read_csv(filepath)
            logger.info(f"✅ {filename}: {len(df)} Artikel geladen")
            success_count += 1
            
        except requests.RequestException as e:
            logger.warning(f"❌ Fehler beim Laden von {filename}: {e}")
            
            # Versuche alternative URL
            alt_url = alt_urls.get(filename)
            if alt_url:
                try:
                    logger.info(f"🔄 Versuche alternative URL für {filename}...")
                    response = requests.get(alt_url, timeout=60)
                    response.raise_for_status()
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    
                    df = pd.read_csv(filepath)
                    logger.info(f"✅ {filename}: {len(df)} Artikel geladen (alternative URL)")
                    success_count += 1
                    
                except Exception as e2:
                    logger.error(f"❌ Auch alternative URL fehlgeschlagen: {e2}")
        
        except Exception as e:
            logger.error(f"❌ Unerwarteter Fehler bei {filename}: {e}")
    
    if success_count == 2:
        logger.info("🎉 ISOT Dataset erfolgreich heruntergeladen!")
        logger.info("🔄 Starten Sie das Backend neu für automatisches Re-Training")
        return True
    else:
        logger.error("❌ ISOT Dataset Download unvollständig")
        return False

def manual_download_instructions():
    """
    Zeigt Anweisungen für manuellen Download
    """
    logger.info("📋 MANUELLE DOWNLOAD-ANWEISUNGEN:")
    logger.info("")
    logger.info("1. Besuchen Sie eine dieser Quellen:")
    logger.info("   • https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    logger.info("   • https://github.com/rexshell/ISOT-Fake-News-Dataset")
    logger.info("   • https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/")
    logger.info("")
    logger.info("2. Laden Sie diese Dateien herunter:")
    logger.info("   • Fake.csv")
    logger.info("   • True.csv")
    logger.info("")
    logger.info("3. Speichern Sie sie in:")
    logger.info(f"   • {Path('datasets').absolute()}/")
    logger.info("")
    logger.info("4. Starten Sie das Backend neu:")
    logger.info("   • python backend/app.py")

if __name__ == "__main__":
    logger.info("🚀 ISOT Dataset Download Utility")
    logger.info("=" * 50)
    
    success = download_isot_dataset()
    
    if not success:
        logger.info("")
        manual_download_instructions()
    
    logger.info("")
    logger.info("📊 Erwartete Performance-Verbesserung:")
    logger.info("  Aktuell (nur LIAR): ~59% Accuracy")
    logger.info("  Mit ISOT: ~85-90% Accuracy")
