import requests
import json
import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class SportsDataAPI:
    """
    A client for the Sportsdata.io API to retrieve player statistics.
    
    Supports multiple sports: NFL, NBA, MLB, NHL, Soccer, etc.
    """
    
    def __init__(self, api_key: str, sport: str = 'nfl'):
        """
        Initialize the API client.
        
        Args:
            api_key: Your Sportsdata.io API subscription key
            sport: Sport type (nfl, nba, mlb, nhl, soccer, etc.)
        """
        self.api_key = api_key
        self.sport = sport.lower()
        self.base_url = f"https://api.sportsdata.io/v3/{self.sport}"
        self.headers = {
            'Ocp-Apim-Subscription-Key': self.api_key
        }
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[Any, Any]:
        """
        Make a GET request to the API.
        
        Args:
            endpoint: API endpoint path
            params: Optional query parameters
            
        Returns:
            JSON response as dictionary
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response: {response.text}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            raise
    
    def get_players(self) -> List[Dict]:
        """
        Get all active players for the sport.
        
        Returns:
            List of player dictionaries
        """
        endpoint = "scores/json/Players"
        return self._make_request(endpoint)
    
    def get_player_season_stats(self, season: str, player_id: Optional[int] = None) -> List[Dict]:
        """
        Get player statistics for a specific season.
        
        Args:
            season: Season year (e.g., "2024", "2024POST" for playoffs)
            player_id: Optional specific player ID
            
        Returns:
            List of player stats dictionaries
        """
        endpoint = f"stats/json/PlayerSeasonStats/{season}"
        return self._make_request(endpoint)
    
    def get_player_game_stats(self, season: str, week: int) -> List[Dict]:
        """
        Get player statistics for a specific week/game (NFL).
        
        Args:
            season: Season year (e.g., "2024")
            week: Week number
            
        Returns:
            List of player game stats
        """
        endpoint = f"stats/json/PlayerGameStatsByWeek/{season}/{week}"
        return self._make_request(endpoint)
    
    def get_player_details(self, player_id: int) -> Dict:
        """
        Get detailed information about a specific player.
        
        Args:
            player_id: The player's ID
            
        Returns:
            Player details dictionary
        """
        endpoint = f"scores/json/Player/{player_id}"
        return self._make_request(endpoint)
    
    def search_players_by_team(self, team: str) -> List[Dict]:
        """
        Get all players for a specific team.
        
        Args:
            team: Team abbreviation (e.g., "KC", "LAL", "NYY")
            
        Returns:
            List of players on the team
        """
        endpoint = f"scores/json/Players/{team}"
        return self._make_request(endpoint)
    
    def get_fantasy_players(self, season: str) -> List[Dict]:
        """
        Get fantasy player information for a season.
        
        Args:
            season: Season year
            
        Returns:
            List of fantasy player data
        """
        endpoint = f"stats/json/FantasyPlayers/{season}"
        return self._make_request(endpoint)
    
    def save_to_file(self, data: Any, filename: str):
        """
        Save API response data to a JSON file.
        
        Args:
            data: Data to save
            filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Load API key from .env file
    API_KEY = os.getenv('SPORTSDATA_API_KEY')
    
    if not API_KEY:
        raise ValueError("SPORTSDATA_API_KEY not found in .env file. Please add it.")
    
    # Initialize the client for NFL
    client = SportsDataAPI(api_key=API_KEY, sport='nfl')
    
    # Example 1: Get all active players
    print("Fetching all active players...")
    try:
        players = client.get_players()
        print(f"Retrieved {len(players)} players")
        
        # Display first 3 players
        for player in players[:3]:
            print(f"- {player.get('FirstName')} {player.get('LastName')} ({player.get('Position')})")
    except Exception as e:
        print(f"Error fetching players: {e}")
    
    # Example 2: Get player stats for current season
    print("\nFetching player season stats for 2024...")
    try:
        season_stats = client.get_player_season_stats("2024")
        print(f"Retrieved stats for {len(season_stats)} players")
        
        # Display stats for first player with passing yards
        for stat in season_stats:
            if stat.get('PassingYards', 0) > 0:
                print(f"- {stat.get('Name')}: {stat.get('PassingYards')} passing yards")
                break
    except Exception as e:
        print(f"Error fetching season stats: {e}")
    
    # Example 3: Get players from a specific team
    print("\nFetching Kansas City Chiefs players...")
    try:
        kc_players = client.search_players_by_team("KC")
        print(f"Retrieved {len(kc_players)} Chiefs players")
    except Exception as e:
        print(f"Error fetching team players: {e}")
    
    # Example 4: Save data to file
    # client.save_to_file(players, "nfl_players.json")
    
    print("\n--- Other Sports Examples ---")
    
    # NBA Example
    # nba_client = SportsDataAPI(api_key=API_KEY, sport='nba')
    # nba_players = nba_client.get_players()
    
    # MLB Example
    # mlb_client = SportsDataAPI(api_key=API_KEY, sport='mlb')
    # mlb_stats = mlb_client.get_player_season_stats("2024")
    
    # Soccer Example
    # soccer_client = SportsDataAPI(api_key=API_KEY, sport='soccer')
    # soccer_players = soccer_client.get_players()
