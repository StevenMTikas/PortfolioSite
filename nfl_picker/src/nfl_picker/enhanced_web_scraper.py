"""
Enhanced web scraper for NFL advanced statistics.
Scrapes data from multiple sources to populate advanced metrics in the stats database.
"""

import requests
import time
import re
import json
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedWebScraper:
    """Enhanced web scraper for NFL advanced statistics."""
    
    def __init__(self, rate_limit: float = 2.0):
        """
        Initialize the enhanced web scraper.
        
        Args:
            rate_limit: Delay between requests in seconds
        """
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Available sources
        self.sources = {
            'espn': 'https://www.espn.com/nfl',
            'nfl': 'https://www.nfl.com',
            'pfr': 'https://www.pro-football-reference.com'
        }
    
    def _make_request(self, url: str, max_retries: int = 3) -> Optional[BeautifulSoup]:
        """Make a request with rate limiting and error handling."""
        for attempt in range(max_retries):
            try:
                time.sleep(self.rate_limit)
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                return BeautifulSoup(response.content, 'html.parser')
            except Exception as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                    return None
    
    def scrape_espn_player_stats(self, position: str, week: Optional[int] = None) -> List[Dict[str, Any]]:
        """Scrape ESPN for player statistics."""
        try:
            # ESPN stats URLs by position
            espn_urls = {
                'QB': f"{self.sources['espn']}/stats/player/_/position/qb",
                'RB': f"{self.sources['espn']}/stats/player/_/position/rb", 
                'WR': f"{self.sources['espn']}/stats/player/_/position/wr",
                'TE': f"{self.sources['espn']}/stats/player/_/position/te",
                'DL': f"{self.sources['espn']}/stats/player/_/position/dl",
                'LB': f"{self.sources['espn']}/stats/player/_/position/lb",
                'CB': f"{self.sources['espn']}/stats/player/_/position/cb",
                'S': f"{self.sources['espn']}/stats/player/_/position/s"
            }
            
            if position not in espn_urls:
                return []
            
            url = espn_urls[position]
            soup = self._make_request(url)
            
            if not soup:
                return []
            
            players = []
            
            # Look for player stats tables
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 4:
                        try:
                            player_data = self._parse_espn_row(cells, position)
                            if player_data:
                                players.append(player_data)
                        except Exception as e:
                            logger.warning(f"Error parsing ESPN row: {e}")
                            continue
            
            logger.info(f"Scraped {len(players)} {position} players from ESPN")
            return players
            
        except Exception as e:
            logger.error(f"Error scraping ESPN {position} stats: {e}")
            return []
    
    def _parse_espn_row(self, cells, position: str) -> Optional[Dict[str, Any]]:
        """Parse a row from ESPN stats table."""
        try:
            if len(cells) < 4:
                return None
            
            # Extract player name (usually first cell)
            name_cell = cells[0]
            player_name = name_cell.get_text().strip()
            
            # Extract team (usually second cell)
            team_cell = cells[1] if len(cells) > 1 else None
            team = team_cell.get_text().strip() if team_cell else "Unknown"
            
            # Position-specific parsing
            if position == 'QB':
                return self._parse_qb_espn_data(cells, player_name, team)
            elif position in ['RB', 'WR', 'TE']:
                return self._parse_skill_position_espn_data(cells, player_name, team, position)
            elif position in ['DL', 'LB', 'CB', 'S']:
                return self._parse_defensive_espn_data(cells, player_name, team, position)
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error parsing ESPN row: {e}")
            return None
    
    def _parse_qb_espn_data(self, cells, player_name: str, team: str) -> Dict[str, Any]:
        """Parse QB data from ESPN."""
        data = {
            'player_name': player_name,
            'team': team,
            'position': 'QB',
            'data_source': 'espn_scraped'
        }
        
        # Map common ESPN QB columns
        if len(cells) >= 8:
            data.update({
                'passing_yards': self._parse_int(cells[2].get_text()) or 0,
                'passing_touchdowns': self._parse_int(cells[3].get_text()) or 0,
                'interceptions': self._parse_int(cells[4].get_text()) or 0,
                'completions': self._parse_int(cells[5].get_text()) or 0,
                'attempts': self._parse_int(cells[6].get_text()) or 0,
                'passer_rating': self._parse_float(cells[7].get_text()) or 0.0
            })
        
        return data
    
    def _parse_skill_position_espn_data(self, cells, player_name: str, team: str, position: str) -> Dict[str, Any]:
        """Parse skill position data from ESPN."""
        data = {
            'player_name': player_name,
            'team': team,
            'position': position,
            'data_source': 'espn_scraped'
        }
        
        if position in ['RB']:
            if len(cells) >= 6:
                data.update({
                    'rushing_yards': self._parse_int(cells[2].get_text()) or 0,
                    'rushing_attempts': self._parse_int(cells[3].get_text()) or 0,
                    'rushing_touchdowns': self._parse_int(cells[4].get_text()) or 0,
                    'receptions': self._parse_int(cells[5].get_text()) or 0
                })
        elif position in ['WR', 'TE']:
            if len(cells) >= 6:
                data.update({
                    'receptions': self._parse_int(cells[2].get_text()) or 0,
                    'receiving_yards': self._parse_int(cells[3].get_text()) or 0,
                    'touchdowns': self._parse_int(cells[4].get_text()) or 0,
                    'targets': self._parse_int(cells[5].get_text()) or 0
                })
        
        return data
    
    def _parse_defensive_espn_data(self, cells, player_name: str, team: str, position: str) -> Dict[str, Any]:
        """Parse defensive data from ESPN."""
        data = {
            'player_name': player_name,
            'team': team,
            'position': position,
            'data_source': 'espn_scraped'
        }
        
        if len(cells) >= 6:
            data.update({
                'tackles': self._parse_int(cells[2].get_text()) or 0,
                'assists': self._parse_int(cells[3].get_text()) or 0,
                'sacks': self._parse_float(cells[4].get_text()) or 0.0,
                'interceptions': self._parse_int(cells[5].get_text()) or 0
            })
        
        return data
    
    def scrape_nfl_com_stats(self, position: str) -> List[Dict[str, Any]]:
        """Scrape NFL.com for player statistics."""
        try:
            # NFL.com stats URLs
            nfl_urls = {
                'QB': f"{self.sources['nfl']}/stats/player-stats/category/passing/2025/reg/all",
                'RB': f"{self.sources['nfl']}/stats/player-stats/category/rushing/2025/reg/all",
                'WR': f"{self.sources['nfl']}/stats/player-stats/category/receiving/2025/reg/all",
                'TE': f"{self.sources['nfl']}/stats/player-stats/category/receiving/2025/reg/all"
            }
            
            if position not in nfl_urls:
                return []
            
            url = nfl_urls[position]
            soup = self._make_request(url)
            
            if not soup:
                return []
            
            players = []
            
            # Look for player stats in NFL.com format
            stats_sections = soup.find_all(['div', 'section'], class_=re.compile(r'stats|player'))
            for section in stats_sections:
                rows = section.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 4:
                        try:
                            player_data = self._parse_nfl_row(cells, position)
                            if player_data:
                                players.append(player_data)
                        except Exception as e:
                            logger.warning(f"Error parsing NFL.com row: {e}")
                            continue
            
            logger.info(f"Scraped {len(players)} {position} players from NFL.com")
            return players
            
        except Exception as e:
            logger.error(f"Error scraping NFL.com {position} stats: {e}")
            return []
    
    def _parse_nfl_row(self, cells, position: str) -> Optional[Dict[str, Any]]:
        """Parse a row from NFL.com stats table."""
        try:
            if len(cells) < 4:
                return None
            
            # Extract player name
            name_cell = cells[0]
            player_name = name_cell.get_text().strip()
            
            # Extract team
            team_cell = cells[1] if len(cells) > 1 else None
            team = team_cell.get_text().strip() if team_cell else "Unknown"
            
            # Create base data
            data = {
                'player_name': player_name,
                'team': team,
                'position': position,
                'data_source': 'nfl_scraped'
            }
            
            # Add position-specific stats
            if position == 'QB':
                if len(cells) >= 8:
                    data.update({
                        'passing_yards': self._parse_int(cells[2].get_text()) or 0,
                        'passing_touchdowns': self._parse_int(cells[3].get_text()) or 0,
                        'interceptions': self._parse_int(cells[4].get_text()) or 0,
                        'completions': self._parse_int(cells[5].get_text()) or 0,
                        'attempts': self._parse_int(cells[6].get_text()) or 0,
                        'passer_rating': self._parse_float(cells[7].get_text()) or 0.0
                    })
            elif position in ['RB']:
                if len(cells) >= 6:
                    data.update({
                        'rushing_yards': self._parse_int(cells[2].get_text()) or 0,
                        'rushing_attempts': self._parse_int(cells[3].get_text()) or 0,
                        'rushing_touchdowns': self._parse_int(cells[4].get_text()) or 0,
                        'receptions': self._parse_int(cells[5].get_text()) or 0
                    })
            elif position in ['WR', 'TE']:
                if len(cells) >= 6:
                    data.update({
                        'receptions': self._parse_int(cells[2].get_text()) or 0,
                        'receiving_yards': self._parse_int(cells[3].get_text()) or 0,
                        'touchdowns': self._parse_int(cells[4].get_text()) or 0,
                        'targets': self._parse_int(cells[5].get_text()) or 0
                    })
            
            return data
            
        except Exception as e:
            logger.warning(f"Error parsing NFL.com row: {e}")
            return None
    
    def scrape_advanced_metrics(self, position: str, week: Optional[int] = None) -> List[Dict[str, Any]]:
        """Scrape advanced metrics from multiple sources."""
        all_data = []
        
        # Try ESPN first
        try:
            espn_data = self.scrape_espn_player_stats(position, week)
            all_data.extend(espn_data)
        except Exception as e:
            logger.warning(f"ESPN scraping failed for {position}: {e}")
        
        # Try NFL.com
        try:
            nfl_data = self.scrape_nfl_com_stats(position)
            all_data.extend(nfl_data)
        except Exception as e:
            logger.warning(f"NFL.com scraping failed for {position}: {e}")
        
        # Remove duplicates based on player name and team
        unique_data = []
        seen = set()
        for player in all_data:
            key = (player.get('player_name', ''), player.get('team', ''))
            if key not in seen:
                seen.add(key)
                unique_data.append(player)
        
        logger.info(f"Collected {len(unique_data)} unique {position} players from all sources")
        return unique_data
    
    def update_database_with_scraped_data(self, position: str, week: int, season: int = 2025) -> int:
        """Update database with scraped data for a position."""
        from nfl_picker.stats_database import NFLStatsDatabase
        
        logger.info(f"Updating {position} with scraped data for week {week}")
        
        try:
            # Get scraped data
            scraped_data = self.scrape_advanced_metrics(position, week)
            
            if not scraped_data:
                logger.warning(f"No scraped data found for {position}")
                return 0
            
            # Connect to database
            db = NFLStatsDatabase()
            
            # Update records
            updated_count = 0
            for player_data in scraped_data:
                try:
                    # Add week and season
                    player_data['week'] = week
                    player_data['season'] = season
                    
                    # Save to appropriate table
                    if position == 'QB':
                        db.save_quarterback_stats(player_data)
                    elif position == 'RB':
                        db.save_running_back_stats(player_data)
                    elif position == 'WR':
                        db.save_wide_receiver_stats(player_data)
                    elif position == 'TE':
                        db.save_tight_end_stats(player_data)
                    elif position == 'DL':
                        db.save_defensive_line_stats(player_data)
                    elif position == 'LB':
                        db.save_linebacker_stats(player_data)
                    elif position == 'CB':
                        db.save_cornerback_stats(player_data)
                    elif position == 'S':
                        db.save_safety_stats(player_data)
                    
                    updated_count += 1
                    
                except Exception as e:
                    logger.error(f"Error saving {position} data for {player_data.get('player_name', 'Unknown')}: {e}")
            
            db.close()
            logger.info(f"Updated {updated_count} {position} records with scraped data")
            return updated_count
            
        except Exception as e:
            logger.error(f"Error updating database with scraped data: {e}")
            return 0
    
    def update_all_positions_with_scraping(self, week: int, season: int = 2025) -> Dict[str, int]:
        """Update all positions with scraped data."""
        positions = ['QB', 'RB', 'WR', 'TE', 'DL', 'LB', 'CB', 'S']
        results = {}
        
        logger.info(f"Starting web scraping update for week {week}, season {season}")
        
        for position in positions:
            try:
                count = self.update_database_with_scraped_data(position, week, season)
                results[position] = count
                logger.info(f"Updated {count} {position} records from web scraping")
            except Exception as e:
                logger.error(f"Failed to update {position} with scraping: {e}")
                results[position] = 0
        
        return results
    
    def _parse_float(self, text: str) -> Optional[float]:
        """Parse float from text."""
        if not text or text.strip() == '':
            return None
        text = re.sub(r'[^\d.-]', '', text.strip())
        try:
            return float(text)
        except ValueError:
            return None
    
    def _parse_int(self, text: str) -> Optional[int]:
        """Parse integer from text."""
        if not text or text.strip() == '':
            return None
        text = re.sub(r'[^\d-]', '', text.strip())
        try:
            return int(text)
        except ValueError:
            return None
    
    def close(self):
        """Close the scraper session."""
        if self.session:
            self.session.close()


def main():
    """Test the enhanced web scraper."""
    scraper = EnhancedWebScraper()
    
    # Test scraping for different positions
    positions = ['QB', 'RB', 'WR']
    
    for position in positions:
        print(f"\n=== Testing {position} Scraping ===")
        data = scraper.scrape_advanced_metrics(position)
        print(f"Found {len(data)} {position} players")
        
        if data:
            print("Sample data:")
            for player in data[:3]:  # Show first 3
                print(f"  {player.get('player_name', 'Unknown')} - {player.get('team', 'Unknown')}")
    
    scraper.close()


if __name__ == "__main__":
    main()
