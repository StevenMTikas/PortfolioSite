"""
API data collector for NFL player statistics.
Integrates with SportsData API to fetch and map player data to database.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from nfl_picker.tools.sportsdata_api_script import SportsDataAPI
from nfl_picker.stats_database import NFLStatsDatabase


class APIDataCollector:
    """Collects NFL player statistics from SportsData API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the API data collector.
        
        Args:
            api_key: SportsData API key (if None, loads from .env)
        """
        if api_key is None:
            api_key = os.getenv('SPORTSDATA_API_KEY')
            if not api_key:
                raise ValueError("SPORTSDATA_API_KEY not found in environment variables")
        
        self.api_client = SportsDataAPI(api_key=api_key, sport='nfl')
        self.stats_db = NFLStatsDatabase()
        
        # Team abbreviation mapping
        self.team_mapping = {
            'ARI': 'Arizona Cardinals',
            'ATL': 'Atlanta Falcons', 
            'BAL': 'Baltimore Ravens',
            'BUF': 'Buffalo Bills',
            'CAR': 'Carolina Panthers',
            'CHI': 'Chicago Bears',
            'CIN': 'Cincinnati Bengals',
            'CLE': 'Cleveland Browns',
            'DAL': 'Dallas Cowboys',
            'DEN': 'Denver Broncos',
            'DET': 'Detroit Lions',
            'GB': 'Green Bay Packers',
            'HOU': 'Houston Texans',
            'IND': 'Indianapolis Colts',
            'JAX': 'Jacksonville Jaguars',
            'KC': 'Kansas City Chiefs',
            'LV': 'Las Vegas Raiders',
            'LAC': 'Los Angeles Chargers',
            'LAR': 'Los Angeles Rams',
            'MIA': 'Miami Dolphins',
            'MIN': 'Minnesota Vikings',
            'NE': 'New England Patriots',
            'NO': 'New Orleans Saints',
            'NYG': 'New York Giants',
            'NYJ': 'New York Jets',
            'PHI': 'Philadelphia Eagles',
            'PIT': 'Pittsburgh Steelers',
            'SF': 'San Francisco 49ers',
            'SEA': 'Seattle Seahawks',
            'TB': 'Tampa Bay Buccaneers',
            'TEN': 'Tennessee Titans',
            'WAS': 'Washington Commanders'
        }
    
    def fetch_team_roster(self, team_abbreviation: str) -> List[Dict]:
        """
        Fetch all players for a specific team.
        
        Args:
            team_abbreviation: Team abbreviation (e.g., 'KC', 'LAR')
            
        Returns:
            List of player dictionaries
        """
        try:
            players = self.api_client.search_players_by_team(team_abbreviation)
            return players
        except Exception as e:
            print(f"Error fetching roster for {team_abbreviation}: {e}")
            return []
    
    def fetch_all_rosters(self) -> Dict[str, List[Dict]]:
        """
        Fetch rosters for all NFL teams.
        
        Returns:
            Dictionary mapping team abbreviations to player lists
        """
        all_rosters = {}
        
        for team_abbr in self.team_mapping.keys():
            print(f"Fetching roster for {team_abbr}...")
            roster = self.fetch_team_roster(team_abbr)
            if roster:
                all_rosters[team_abbr] = roster
                print(f"Retrieved {len(roster)} players for {team_abbr}")
            else:
                print(f"No players found for {team_abbr}")
        
        return all_rosters
    
    def map_api_to_db(self, api_data: Dict, position: str) -> Dict[str, Any]:
        """
        Map API data to database format for a specific position.
        
        Args:
            api_data: Raw API data for a player
            position: Player position
            
        Returns:
            Mapped data dictionary
        """
        base_data = {
            'player_id': api_data.get('PlayerID'),
            'player_name': f"{api_data.get('FirstName', '')} {api_data.get('LastName', '')}".strip(),
            'team': self.team_mapping.get(api_data.get('Team', ''), api_data.get('Team', '')),
            'position': position,
            'data_source': 'api'
        }
        
        # Position-specific mapping
        if position in ['DL', 'DE', 'DT', 'NT']:
            return self._map_defensive_line_data(api_data, base_data)
        elif position in ['LB', 'ILB', 'OLB', 'MLB']:
            return self._map_linebacker_data(api_data, base_data)
        elif position in ['CB', 'DB']:
            return self._map_cornerback_data(api_data, base_data)
        elif position in ['S', 'FS', 'SS']:
            return self._map_safety_data(api_data, base_data)
        elif position in ['OL', 'OT', 'OG', 'C']:
            return self._map_offensive_line_data(api_data, base_data)
        elif position == 'TE':
            return self._map_tight_end_data(api_data, base_data)
        elif position == 'WR':
            return self._map_wide_receiver_data(api_data, base_data)
        elif position == 'RB':
            return self._map_running_back_data(api_data, base_data)
        elif position == 'QB':
            return self._map_quarterback_data(api_data, base_data)
        elif position in ['K', 'P', 'KR', 'PR']:
            return self._map_special_teams_data(api_data, base_data)
        else:
            return base_data
    
    def _map_defensive_line_data(self, api_data: Dict, base_data: Dict) -> Dict[str, Any]:
        """Map defensive line specific data."""
        return {
            **base_data,
            'tackles': api_data.get('Tackles', 0),
            'assists': api_data.get('Assists', 0),
            'sacks': api_data.get('Sacks', 0),
            'fumbles_forced': api_data.get('FumblesForced', 0),
            'fumbles_recovered': api_data.get('FumblesRecovered', 0)
        }
    
    def _map_linebacker_data(self, api_data: Dict, base_data: Dict) -> Dict[str, Any]:
        """Map linebacker specific data."""
        return {
            **base_data,
            'tackles': api_data.get('Tackles', 0),
            'assists': api_data.get('Assists', 0),
            'sacks': api_data.get('Sacks', 0),
            'interceptions': api_data.get('Interceptions', 0),
            'passes_defended': api_data.get('PassesDefended', 0)
        }
    
    def _map_cornerback_data(self, api_data: Dict, base_data: Dict) -> Dict[str, Any]:
        """Map cornerback specific data."""
        return {
            **base_data,
            'tackles': api_data.get('Tackles', 0),
            'assists': api_data.get('Assists', 0),
            'interceptions': api_data.get('Interceptions', 0),
            'passes_defended': api_data.get('PassesDefended', 0)
        }
    
    def _map_safety_data(self, api_data: Dict, base_data: Dict) -> Dict[str, Any]:
        """Map safety specific data."""
        return {
            **base_data,
            'tackles': api_data.get('Tackles', 0),
            'assists': api_data.get('Assists', 0),
            'interceptions': api_data.get('Interceptions', 0),
            'passes_defended': api_data.get('PassesDefended', 0),
            'sacks': api_data.get('Sacks', 0)
        }
    
    def _map_offensive_line_data(self, api_data: Dict, base_data: Dict) -> Dict[str, Any]:
        """Map offensive line specific data."""
        return {
            **base_data,
            'games_played': api_data.get('GamesPlayed', 0),
            'games_started': api_data.get('GamesStarted', 0)
        }
    
    def _map_tight_end_data(self, api_data: Dict, base_data: Dict) -> Dict[str, Any]:
        """Map tight end specific data."""
        return {
            **base_data,
            'receptions': api_data.get('Receptions', 0),
            'receiving_yards': api_data.get('ReceivingYards', 0),
            'touchdowns': api_data.get('ReceivingTouchdowns', 0),
            'targets': api_data.get('Targets', 0),
            'yards_per_reception': api_data.get('ReceivingYards', 0) / max(api_data.get('Receptions', 1), 1)
        }
    
    def _map_wide_receiver_data(self, api_data: Dict, base_data: Dict) -> Dict[str, Any]:
        """Map wide receiver specific data."""
        return {
            **base_data,
            'receptions': api_data.get('Receptions', 0),
            'receiving_yards': api_data.get('ReceivingYards', 0),
            'touchdowns': api_data.get('ReceivingTouchdowns', 0),
            'targets': api_data.get('Targets', 0),
            'yards_per_reception': api_data.get('ReceivingYards', 0) / max(api_data.get('Receptions', 1), 1)
        }
    
    def _map_running_back_data(self, api_data: Dict, base_data: Dict) -> Dict[str, Any]:
        """Map running back specific data."""
        return {
            **base_data,
            'rushing_yards': api_data.get('RushingYards', 0),
            'rushing_attempts': api_data.get('RushingAttempts', 0),
            'rushing_touchdowns': api_data.get('RushingTouchdowns', 0),
            'receptions': api_data.get('Receptions', 0),
            'receiving_yards': api_data.get('ReceivingYards', 0),
            'receiving_touchdowns': api_data.get('ReceivingTouchdowns', 0),
            'fumbles': api_data.get('Fumbles', 0)
        }
    
    def _map_quarterback_data(self, api_data: Dict, base_data: Dict) -> Dict[str, Any]:
        """Map quarterback specific data."""
        return {
            **base_data,
            'passing_yards': api_data.get('PassingYards', 0),
            'passing_touchdowns': api_data.get('PassingTouchdowns', 0),
            'interceptions': api_data.get('Interceptions', 0),
            'completions': api_data.get('Completions', 0),
            'attempts': api_data.get('PassingAttempts', 0),
            'passer_rating': api_data.get('PasserRating', 0)
        }
    
    def _map_special_teams_data(self, api_data: Dict, base_data: Dict) -> Dict[str, Any]:
        """Map special teams specific data."""
        return {
            **base_data,
            'field_goals_made': api_data.get('FieldGoalsMade', 0),
            'field_goals_attempted': api_data.get('FieldGoalsAttempted', 0),
            'extra_points_made': api_data.get('ExtraPointsMade', 0),
            'extra_points_attempted': api_data.get('ExtraPointsAttempted', 0),
            'punts': api_data.get('Punts', 0),
            'punt_yards': api_data.get('PuntYards', 0)
        }
    
    def populate_position_table(self, position: str, week: int, season: int = 2025) -> int:
        """
        Populate database with data for a specific position.
        
        Args:
            position: Position to populate
            week: Week number
            season: Season year
            
        Returns:
            Number of players processed
        """
        position_mapping = {
            'DL': ['DL', 'DE', 'DT', 'NT'],
            'LB': ['LB', 'ILB', 'OLB', 'MLB'],
            'CB': ['CB', 'DB'],
            'S': ['S', 'FS', 'SS'],
            'OL': ['OL', 'OT', 'OG', 'C'],
            'TE': ['TE'],
            'WR': ['WR'],
            'RB': ['RB'],
            'QB': ['QB'],
            'ST': ['K', 'P', 'KR', 'PR']
        }
        
        if position not in position_mapping:
            print(f"Invalid position: {position}")
            return 0
        
        target_positions = position_mapping[position]
        players_processed = 0
        
        # Get all team rosters
        all_rosters = self.fetch_all_rosters()
        
        for team_abbr, roster in all_rosters.items():
            for player in roster:
                player_position = player.get('Position', '')
                
                if player_position in target_positions:
                    # Map API data to database format
                    mapped_data = self.map_api_to_db(player, position)
                    mapped_data['week'] = week
                    mapped_data['season'] = season
                    
                    # Save to appropriate table
                    try:
                        if position == 'DL':
                            self.stats_db.save_defensive_line_stats(mapped_data)
                        elif position == 'LB':
                            self.stats_db.save_linebacker_stats(mapped_data)
                        elif position == 'CB':
                            self.stats_db.save_cornerback_stats(mapped_data)
                        elif position == 'S':
                            self.stats_db.save_safety_stats(mapped_data)
                        elif position == 'OL':
                            self.stats_db.save_offensive_line_stats(mapped_data)
                        elif position == 'TE':
                            self.stats_db.save_tight_end_stats(mapped_data)
                        elif position == 'WR':
                            self.stats_db.save_wide_receiver_stats(mapped_data)
                        elif position == 'RB':
                            self.stats_db.save_running_back_stats(mapped_data)
                        elif position == 'QB':
                            self.stats_db.save_quarterback_stats(mapped_data)
                        elif position == 'ST':
                            self.stats_db.save_special_teams_stats(mapped_data)
                        
                        players_processed += 1
                        
                    except Exception as e:
                        print(f"Error saving {position} data for {player.get('FirstName', '')} {player.get('LastName', '')}: {e}")
        
        print(f"Processed {players_processed} {position} players for week {week}")
        return players_processed
    
    def update_all_positions(self, week: int, season: int = 2025) -> Dict[str, int]:
        """
        Update all position tables with current data.
        
        Args:
            week: Week number
            season: Season year
            
        Returns:
            Dictionary with position counts
        """
        positions = ['DL', 'LB', 'CB', 'S', 'OL', 'TE', 'WR', 'RB', 'QB', 'ST']
        results = {}
        
        print(f"Updating all position data for week {week}, season {season}")
        
        for position in positions:
            print(f"Processing {position}...")
            count = self.populate_position_table(position, week, season)
            results[position] = count
        
        return results
    
    def save_team_metadata(self, team_abbreviation: str, team_name: str, 
                          division: str = None, conference: str = None) -> None:
        """
        Save team metadata to database.
        
        Args:
            team_abbreviation: Team abbreviation
            team_name: Full team name
            division: Division name
            conference: Conference name
        """
        team_data = {
            'team_name': team_name,
            'team_abbreviation': team_abbreviation,
            'division': division,
            'conference': conference,
            'last_api_update': datetime.now().isoformat(),
            'data_source': 'api'
        }
        
        self.stats_db.save_team_metadata(team_data)
    
    def close(self):
        """Close database connection."""
        self.stats_db.close()
