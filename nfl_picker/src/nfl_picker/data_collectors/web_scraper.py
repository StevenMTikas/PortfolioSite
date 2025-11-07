"""
Web scraper for advanced NFL statistics.
Scrapes data from Pro Football Reference, ESPN, and other sources for advanced metrics.
"""

import requests
import time
import re
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraper:
    """Web scraper for advanced NFL statistics not available in free API."""
    
    def __init__(self, rate_limit: float = 1.0):
        """
        Initialize the web scraper.
        
        Args:
            rate_limit: Delay between requests in seconds
        """
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Base URLs for different sources
        self.pfr_base = "https://www.pro-football-reference.com"
        self.espn_base = "https://www.espn.com/nfl"
        self.nfl_base = "https://www.nfl.com"
    
    def _make_request(self, url: str, max_retries: int = 3) -> Optional[BeautifulSoup]:
        """
        Make a request with rate limiting and error handling.
        
        Args:
            url: URL to request
            max_retries: Maximum number of retry attempts
            
        Returns:
            BeautifulSoup object or None if failed
        """
        for attempt in range(max_retries):
            try:
                time.sleep(self.rate_limit)
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return BeautifulSoup(response.content, 'html.parser')
            except Exception as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                    return None
    
    def scrape_pff_grades(self, position: str, week: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Scrape Pro Football Focus grades for a position.
        
        Args:
            position: Position to scrape (e.g., 'QB', 'WR', 'CB')
            week: Specific week (None for season totals)
            
        Returns:
            List of player grade dictionaries
        """
        # Note: PFF grades are typically behind a paywall
        # This is a placeholder for when/if we get access
        logger.info(f"PFF grades scraping not implemented for {position}")
        return []
    
    def scrape_advanced_metrics(self, position: str, week: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Scrape advanced metrics from Pro Football Reference.
        
        Args:
            position: Position to scrape
            week: Specific week (None for season totals)
            
        Returns:
            List of advanced metrics dictionaries
        """
        if position == 'QB':
            return self._scrape_qb_advanced_metrics(week)
        elif position in ['WR', 'TE', 'RB']:
            return self._scrape_skill_position_metrics(position, week)
        elif position in ['DL', 'LB', 'CB', 'S']:
            return self._scrape_defensive_metrics(position, week)
        else:
            logger.warning(f"Advanced metrics scraping not implemented for {position}")
            return []
    
    def _scrape_qb_advanced_metrics(self, week: Optional[int] = None) -> List[Dict[str, Any]]:
        """Scrape advanced QB metrics from Pro Football Reference."""
        url = f"{self.pfr_base}/leaders/pass/pass_rating/2025.htm"
        soup = self._make_request(url)
        
        if not soup:
            return []
        
        qb_data = []
        try:
            # Find the passing leaders table
            table = soup.find('table', {'id': 'passing'})
            if not table:
                return []
            
            rows = table.find('tbody').find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 8:
                    player_data = {
                        'player_name': cells[0].get_text().strip(),
                        'team': cells[1].get_text().strip(),
                        'completion_pct': self._parse_float(cells[3].get_text()),
                        'yards_per_attempt': self._parse_float(cells[4].get_text()),
                        'touchdown_rate': self._parse_float(cells[5].get_text()),
                        'interception_rate': self._parse_float(cells[6].get_text()),
                        'passer_rating': self._parse_float(cells[7].get_text()),
                        'data_source': 'pfr_scraped'
                    }
                    qb_data.append(player_data)
        except Exception as e:
            logger.error(f"Error scraping QB metrics: {e}")
        
        return qb_data
    
    def _scrape_skill_position_metrics(self, position: str, week: Optional[int] = None) -> List[Dict[str, Any]]:
        """Scrape advanced metrics for skill positions."""
        if position == 'WR':
            url = f"{self.pfr_base}/leaders/rec/rec_yds/2025.htm"
        elif position == 'RB':
            url = f"{self.pfr_base}/leaders/rush/rush_yds/2025.htm"
        elif position == 'TE':
            url = f"{self.pfr_base}/leaders/rec/rec_yds/2025.htm"
        else:
            return []
        
        soup = self._make_request(url)
        if not soup:
            return []
        
        player_data = []
        try:
            # Find the appropriate table
            table_id = 'receiving' if position in ['WR', 'TE'] else 'rushing'
            table = soup.find('table', {'id': table_id})
            if not table:
                return []
            
            rows = table.find('tbody').find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 6:
                    data = {
                        'player_name': cells[0].get_text().strip(),
                        'team': cells[1].get_text().strip(),
                        'data_source': 'pfr_scraped'
                    }
                    
                    if position in ['WR', 'TE']:
                        data.update({
                            'receptions': self._parse_int(cells[2].get_text()),
                            'receiving_yards': self._parse_int(cells[3].get_text()),
                            'yards_per_reception': self._parse_float(cells[4].get_text()),
                            'touchdowns': self._parse_int(cells[5].get_text())
                        })
                    elif position == 'RB':
                        data.update({
                            'rushing_attempts': self._parse_int(cells[2].get_text()),
                            'rushing_yards': self._parse_int(cells[3].get_text()),
                            'yards_per_carry': self._parse_float(cells[4].get_text()),
                            'touchdowns': self._parse_int(cells[5].get_text())
                        })
                    
                    player_data.append(data)
        except Exception as e:
            logger.error(f"Error scraping {position} metrics: {e}")
        
        return player_data
    
    def _scrape_defensive_metrics(self, position: str, week: Optional[int] = None) -> List[Dict[str, Any]]:
        """Scrape advanced defensive metrics."""
        url = f"{self.pfr_base}/leaders/def/def_int/2025.htm"
        soup = self._make_request(url)
        
        if not soup:
            return []
        
        defensive_data = []
        try:
            # Find defensive stats table
            table = soup.find('table', {'id': 'defense'})
            if not table:
                return []
            
            rows = table.find('tbody').find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 8:
                    data = {
                        'player_name': cells[0].get_text().strip(),
                        'team': cells[1].get_text().strip(),
                        'position': cells[2].get_text().strip(),
                        'tackles': self._parse_int(cells[3].get_text()),
                        'assists': self._parse_int(cells[4].get_text()),
                        'sacks': self._parse_float(cells[5].get_text()),
                        'interceptions': self._parse_int(cells[6].get_text()),
                        'passes_defended': self._parse_int(cells[7].get_text()),
                        'data_source': 'pfr_scraped'
                    }
                    defensive_data.append(data)
        except Exception as e:
            logger.error(f"Error scraping defensive metrics: {e}")
        
        return defensive_data
    
    def scrape_team_stats(self, team_abbreviation: str) -> Dict[str, Any]:
        """
        Scrape team-level statistics.
        
        Args:
            team_abbreviation: Team abbreviation (e.g., 'KC', 'LAR')
            
        Returns:
            Dictionary of team statistics
        """
        # This would scrape team stats from various sources
        # For now, return placeholder data
        logger.info(f"Team stats scraping not fully implemented for {team_abbreviation}")
        return {
            'team': team_abbreviation,
            'data_source': 'scraped',
            'last_updated': time.time()
        }
    
    def update_scraped_data(self, position: str, week: int, season: int = 2025) -> int:
        """
        Update database with scraped data for a position.
        
        Args:
            position: Position to update
            week: Week number
            season: Season year
            
        Returns:
            Number of records updated
        """
        logger.info(f"Updating scraped data for {position}, week {week}")
        
        # Get advanced metrics
        advanced_data = self.scrape_advanced_metrics(position, week)
        
        # This would integrate with the stats database
        # For now, just log the data
        logger.info(f"Scraped {len(advanced_data)} records for {position}")
        
        return len(advanced_data)
    
    def _parse_float(self, text: str) -> Optional[float]:
        """Parse float from text, handling common formatting."""
        if not text or text.strip() == '':
            return None
        
        # Remove common suffixes and clean text
        text = re.sub(r'[^\d.-]', '', text.strip())
        try:
            return float(text)
        except ValueError:
            return None
    
    def _parse_int(self, text: str) -> Optional[int]:
        """Parse integer from text."""
        if not text or text.strip() == '':
            return None
        
        # Remove non-numeric characters except minus sign
        text = re.sub(r'[^\d-]', '', text.strip())
        try:
            return int(text)
        except ValueError:
            return None
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources."""
        return [
            'Pro Football Reference',
            'ESPN (limited)',
            'NFL.com (limited)',
            'PFF (paywall)'
        ]
    
    def validate_scraping_targets(self) -> Dict[str, bool]:
        """
        Validate that scraping targets are accessible.
        
        Returns:
            Dictionary of source accessibility
        """
        targets = {
            'pfr': False,
            'espn': False,
            'nfl': False
        }
        
        # Test PFR
        try:
            soup = self._make_request(f"{self.pfr_base}/leaders/pass/pass_rating/2025.htm")
            targets['pfr'] = soup is not None
        except:
            pass
        
        # Test ESPN
        try:
            soup = self._make_request(f"{self.espn_base}/stats/player")
            targets['espn'] = soup is not None
        except:
            pass
        
        # Test NFL.com
        try:
            soup = self._make_request(f"{self.nfl_base}/stats/")
            targets['nfl'] = soup is not None
        except:
            pass
        
        return targets
    
    def close(self):
        """Close the scraper session."""
        if self.session:
            self.session.close()
