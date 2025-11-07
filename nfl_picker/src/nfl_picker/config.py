"""
Configuration file for NFL Picker system.
Loads environment variables and provides configuration settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

# API Keys
SERPER_API_KEY = os.getenv('SERPER_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
SPORTSDATA_API_KEY = os.getenv('SPORTSDATA_API_KEY', '')

# NFL Teams List (alphabetically sorted)
NFL_TEAMS = [
    "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
    "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
    "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
    "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
    "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
    "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
    "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
    "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders"
]

# Stats Database Configuration
STATS_DB_PATH = os.getenv('STATS_DB_PATH', 'data/nfl_stats.db')
UPDATE_DAY = os.getenv('UPDATE_DAY', 'Tuesday')
SCRAPING_ENABLED = os.getenv('SCRAPING_ENABLED', 'true').lower() == 'true'

# Configuration settings
CONFIG = {
    'serper_api_key': SERPER_API_KEY,
    'openai_api_key': OPENAI_API_KEY,
    'openai_base_url': OPENAI_BASE_URL,
    'openai_model': OPENAI_MODEL,
    'sportsdata_api_key': SPORTSDATA_API_KEY,
    'stats_db_path': STATS_DB_PATH,
    'update_day': UPDATE_DAY,
    'scraping_enabled': SCRAPING_ENABLED,
    'current_year': 2025,
    'nfl_season': '2025 NFL Season',
}
