"""
Statistics database manager for NFL player position data.
Handles all database operations for position-specific player statistics.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime


class NFLStatsDatabase:
    """Database manager for NFL player statistics by position."""
    
    def __init__(self, db_path: str = "data/nfl_stats.db"):
        """
        Initialize database connection and create tables if needed.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
        """Create all position-specific tables if they don't exist."""
        
        # Defensive Line Stats
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS defensive_line_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                week INTEGER NOT NULL,
                season INTEGER NOT NULL,
                -- Free API stats
                tackles INTEGER DEFAULT 0,
                assists INTEGER DEFAULT 0,
                sacks REAL DEFAULT 0,
                fumbles_forced INTEGER DEFAULT 0,
                fumbles_recovered INTEGER DEFAULT 0,
                -- Web scraping stats (NULL if not available)
                pass_rush_win_rate REAL,
                run_stop_rate REAL,
                pressure_rate REAL,
                block_shedding REAL,
                stuns_defeats_blocks REAL,
                penetration_pursuits REAL,
                -- Metadata
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'api',
                UNIQUE(player_id, week, season)
            )
        ''')
        
        # Linebacker Stats
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS linebacker_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                week INTEGER NOT NULL,
                season INTEGER NOT NULL,
                -- Free API stats
                tackles INTEGER DEFAULT 0,
                assists INTEGER DEFAULT 0,
                sacks REAL DEFAULT 0,
                interceptions INTEGER DEFAULT 0,
                passes_defended INTEGER DEFAULT 0,
                -- Web scraping stats
                tackle_probability REAL,
                run_stop_percentage REAL,
                pressures INTEGER,
                completion_pct_allowed REAL,
                forced_incompletions INTEGER,
                coverage_grade REAL,
                epa REAL,
                dvoa REAL,
                defensive_stops INTEGER,
                -- Metadata
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'api',
                UNIQUE(player_id, week, season)
            )
        ''')
        
        # Cornerback Stats
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS cornerback_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                week INTEGER NOT NULL,
                season INTEGER NOT NULL,
                -- Free API stats
                tackles INTEGER DEFAULT 0,
                assists INTEGER DEFAULT 0,
                interceptions INTEGER DEFAULT 0,
                passes_defended INTEGER DEFAULT 0,
                -- Web scraping stats
                completion_pct_allowed REAL,
                passer_rating_allowed REAL,
                target_rate REAL,
                forced_incompletions INTEGER,
                interception_rate REAL,
                yards_per_target REAL,
                coverage_grade REAL,
                tackle_efficiency REAL,
                press_coverage_success_rate REAL,
                route_recognition REAL,
                -- Metadata
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'api',
                UNIQUE(player_id, week, season)
            )
        ''')
        
        # Safety Stats
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS safety_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                week INTEGER NOT NULL,
                season INTEGER NOT NULL,
                -- Free API stats
                tackles INTEGER DEFAULT 0,
                assists INTEGER DEFAULT 0,
                interceptions INTEGER DEFAULT 0,
                passes_defended INTEGER DEFAULT 0,
                sacks REAL DEFAULT 0,
                -- Web scraping stats
                coverage_grade REAL,
                completion_pct_allowed REAL,
                passer_rating_allowed REAL,
                interception_rate REAL,
                tackle_efficiency REAL,
                missed_tackle_rate REAL,
                run_stop_percentage REAL,
                deep_ball_coverage REAL,
                blitz_effectiveness REAL,
                communication_leadership REAL,
                -- Metadata
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'api',
                UNIQUE(player_id, week, season)
            )
        ''')
        
        # Offensive Line Stats
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS offensive_line_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                week INTEGER NOT NULL,
                season INTEGER NOT NULL,
                -- Free API stats
                games_played INTEGER DEFAULT 0,
                games_started INTEGER DEFAULT 0,
                -- Web scraping stats
                pass_block_win_rate REAL,
                run_block_win_rate REAL,
                pressure_rate_allowed REAL,
                sack_rate_allowed REAL,
                hurry_rate_allowed REAL,
                hit_rate_allowed REAL,
                run_blocking_grade REAL,
                pass_blocking_grade REAL,
                penalty_rate REAL,
                snap_count INTEGER,
                durability REAL,
                -- Metadata
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'api',
                UNIQUE(player_id, week, season)
            )
        ''')
        
        # Tight End Stats
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tight_end_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                week INTEGER NOT NULL,
                season INTEGER NOT NULL,
                -- Free API stats
                receptions INTEGER DEFAULT 0,
                receiving_yards INTEGER DEFAULT 0,
                touchdowns INTEGER DEFAULT 0,
                targets INTEGER DEFAULT 0,
                yards_per_reception REAL DEFAULT 0,
                -- Web scraping stats
                catch_rate REAL,
                target_share REAL,
                route_running_grade REAL,
                blocking_grade REAL,
                red_zone_efficiency REAL,
                third_down_conversion_rate REAL,
                -- Metadata
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'api',
                UNIQUE(player_id, week, season)
            )
        ''')
        
        # Wide Receiver Stats
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS wide_receiver_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                week INTEGER NOT NULL,
                season INTEGER NOT NULL,
                -- Free API stats
                receptions INTEGER DEFAULT 0,
                receiving_yards INTEGER DEFAULT 0,
                touchdowns INTEGER DEFAULT 0,
                targets INTEGER DEFAULT 0,
                yards_per_reception REAL DEFAULT 0,
                -- Web scraping stats
                catch_rate REAL,
                target_share REAL,
                route_running_grade REAL,
                separation REAL,
                contested_catch_rate REAL,
                yards_after_catch REAL,
                red_zone_efficiency REAL,
                -- Metadata
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'api',
                UNIQUE(player_id, week, season)
            )
        ''')
        
        # Running Back Stats
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS running_back_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                week INTEGER NOT NULL,
                season INTEGER NOT NULL,
                -- Free API stats
                rushing_yards INTEGER DEFAULT 0,
                rushing_attempts INTEGER DEFAULT 0,
                rushing_touchdowns INTEGER DEFAULT 0,
                receptions INTEGER DEFAULT 0,
                receiving_yards INTEGER DEFAULT 0,
                receiving_touchdowns INTEGER DEFAULT 0,
                fumbles INTEGER DEFAULT 0,
                -- Web scraping stats
                yards_per_carry REAL,
                rushing_grade REAL,
                receiving_grade REAL,
                pass_blocking_grade REAL,
                fumble_rate REAL,
                breakaway_run_rate REAL,
                -- Metadata
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'api',
                UNIQUE(player_id, week, season)
            )
        ''')
        
        # Quarterback Stats
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quarterback_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                week INTEGER NOT NULL,
                season INTEGER NOT NULL,
                -- Free API stats
                passing_yards INTEGER DEFAULT 0,
                passing_touchdowns INTEGER DEFAULT 0,
                interceptions INTEGER DEFAULT 0,
                completions INTEGER DEFAULT 0,
                attempts INTEGER DEFAULT 0,
                passer_rating REAL DEFAULT 0,
                -- Web scraping stats
                qbr REAL,
                epa REAL,
                cpoe REAL,
                pocket_presence_grade REAL,
                deep_ball_accuracy REAL,
                red_zone_efficiency REAL,
                mobility_rushing_yards INTEGER,
                mobility_rushing_tds INTEGER,
                -- Metadata
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'api',
                UNIQUE(player_id, week, season)
            )
        ''')
        
        # Special Teams Stats
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS special_teams_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                week INTEGER NOT NULL,
                season INTEGER NOT NULL,
                -- Free API stats
                field_goals_made INTEGER DEFAULT 0,
                field_goals_attempted INTEGER DEFAULT 0,
                extra_points_made INTEGER DEFAULT 0,
                extra_points_attempted INTEGER DEFAULT 0,
                punts INTEGER DEFAULT 0,
                punt_yards INTEGER DEFAULT 0,
                -- Web scraping stats
                kick_return_avg REAL,
                punt_return_avg REAL,
                touchback_rate REAL,
                inside_20_percentage REAL,
                hang_time REAL,
                return_touchdowns INTEGER,
                -- Metadata
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'api',
                UNIQUE(player_id, week, season)
            )
        ''')
        
        # Team Metadata
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT NOT NULL,
                team_abbreviation TEXT NOT NULL,
                division TEXT,
                conference TEXT,
                last_api_update TIMESTAMP,
                data_source TEXT DEFAULT 'api',
                UNIQUE(team_abbreviation)
            )
        ''')
        
        self.conn.commit()
    
    # Defensive Line Methods
    def save_defensive_line_stats(self, player_data: Dict[str, Any]) -> None:
        """Save defensive line player statistics."""
        self.cursor.execute("""
            INSERT OR REPLACE INTO defensive_line_stats 
            (player_id, player_name, team, position, week, season, tackles, assists, sacks, 
             fumbles_forced, fumbles_recovered, pass_rush_win_rate, run_stop_rate, pressure_rate, 
             block_shedding, stuns_defeats_blocks, penetration_pursuits, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_data.get('player_id'),
            player_data.get('player_name'),
            player_data.get('team'),
            player_data.get('position'),
            player_data.get('week'),
            player_data.get('season'),
            player_data.get('tackles', 0),
            player_data.get('assists', 0),
            player_data.get('sacks', 0),
            player_data.get('fumbles_forced', 0),
            player_data.get('fumbles_recovered', 0),
            player_data.get('pass_rush_win_rate'),
            player_data.get('run_stop_rate'),
            player_data.get('pressure_rate'),
            player_data.get('block_shedding'),
            player_data.get('stuns_defeats_blocks'),
            player_data.get('penetration_pursuits'),
            player_data.get('data_source', 'api')
        ))
        self.conn.commit()
    
    def get_defensive_line_stats(self, team: Optional[str] = None, player_id: Optional[int] = None, 
                                week: Optional[int] = None, season: int = 2025) -> List[Dict]:
        """Get defensive line statistics."""
        query = "SELECT * FROM defensive_line_stats WHERE season = ?"
        params = [season]
        
        if team:
            query += " AND team = ?"
            params.append(team)
        if player_id:
            query += " AND player_id = ?"
            params.append(player_id)
        if week:
            query += " AND week = ?"
            params.append(week)
        
        query += " ORDER BY tackles DESC"
        
        self.cursor.execute(query, params)
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    # Linebacker Methods
    def save_linebacker_stats(self, player_data: Dict[str, Any]) -> None:
        """Save linebacker player statistics."""
        self.cursor.execute("""
            INSERT OR REPLACE INTO linebacker_stats 
            (player_id, player_name, team, position, week, season, tackles, assists, sacks, 
             interceptions, passes_defended, tackle_probability, run_stop_percentage, pressures, 
             completion_pct_allowed, forced_incompletions, coverage_grade, epa, dvoa, defensive_stops, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_data.get('player_id'),
            player_data.get('player_name'),
            player_data.get('team'),
            player_data.get('position'),
            player_data.get('week'),
            player_data.get('season'),
            player_data.get('tackles', 0),
            player_data.get('assists', 0),
            player_data.get('sacks', 0),
            player_data.get('interceptions', 0),
            player_data.get('passes_defended', 0),
            player_data.get('tackle_probability'),
            player_data.get('run_stop_percentage'),
            player_data.get('pressures'),
            player_data.get('completion_pct_allowed'),
            player_data.get('forced_incompletions'),
            player_data.get('coverage_grade'),
            player_data.get('epa'),
            player_data.get('dvoa'),
            player_data.get('defensive_stops'),
            player_data.get('data_source', 'api')
        ))
        self.conn.commit()
    
    def get_linebacker_stats(self, team: Optional[str] = None, player_id: Optional[int] = None, 
                            week: Optional[int] = None, season: int = 2025) -> List[Dict]:
        """Get linebacker statistics."""
        query = "SELECT * FROM linebacker_stats WHERE season = ?"
        params = [season]
        
        if team:
            query += " AND team = ?"
            params.append(team)
        if player_id:
            query += " AND player_id = ?"
            params.append(player_id)
        if week:
            query += " AND week = ?"
            params.append(week)
        
        query += " ORDER BY tackles DESC"
        
        self.cursor.execute(query, params)
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    # Cornerback Methods
    def save_cornerback_stats(self, player_data: Dict[str, Any]) -> None:
        """Save cornerback player statistics."""
        self.cursor.execute("""
            INSERT OR REPLACE INTO cornerback_stats 
            (player_id, player_name, team, position, week, season, tackles, assists, interceptions, 
             passes_defended, completion_pct_allowed, passer_rating_allowed, target_rate, 
             forced_incompletions, interception_rate, yards_per_target, coverage_grade, 
             tackle_efficiency, press_coverage_success_rate, route_recognition, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_data.get('player_id'),
            player_data.get('player_name'),
            player_data.get('team'),
            player_data.get('position'),
            player_data.get('week'),
            player_data.get('season'),
            player_data.get('tackles', 0),
            player_data.get('assists', 0),
            player_data.get('interceptions', 0),
            player_data.get('passes_defended', 0),
            player_data.get('completion_pct_allowed'),
            player_data.get('passer_rating_allowed'),
            player_data.get('target_rate'),
            player_data.get('forced_incompletions'),
            player_data.get('interception_rate'),
            player_data.get('yards_per_target'),
            player_data.get('coverage_grade'),
            player_data.get('tackle_efficiency'),
            player_data.get('press_coverage_success_rate'),
            player_data.get('route_recognition'),
            player_data.get('data_source', 'api')
        ))
        self.conn.commit()
    
    def get_cornerback_stats(self, team: Optional[str] = None, player_id: Optional[int] = None, 
                           week: Optional[int] = None, season: int = 2025) -> List[Dict]:
        """Get cornerback statistics."""
        query = "SELECT * FROM cornerback_stats WHERE season = ?"
        params = [season]
        
        if team:
            query += " AND team = ?"
            params.append(team)
        if player_id:
            query += " AND player_id = ?"
            params.append(player_id)
        if week:
            query += " AND week = ?"
            params.append(week)
        
        query += " ORDER BY interceptions DESC, tackles DESC"
        
        self.cursor.execute(query, params)
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    # Safety Methods
    def save_safety_stats(self, player_data: Dict[str, Any]) -> None:
        """Save safety player statistics."""
        self.cursor.execute("""
            INSERT OR REPLACE INTO safety_stats 
            (player_id, player_name, team, position, week, season, tackles, assists, interceptions, 
             passes_defended, sacks, coverage_grade, completion_pct_allowed, passer_rating_allowed, 
             interception_rate, tackle_efficiency, missed_tackle_rate, run_stop_percentage, 
             deep_ball_coverage, blitz_effectiveness, communication_leadership, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_data.get('player_id'),
            player_data.get('player_name'),
            player_data.get('team'),
            player_data.get('position'),
            player_data.get('week'),
            player_data.get('season'),
            player_data.get('tackles', 0),
            player_data.get('assists', 0),
            player_data.get('interceptions', 0),
            player_data.get('passes_defended', 0),
            player_data.get('sacks', 0),
            player_data.get('coverage_grade'),
            player_data.get('completion_pct_allowed'),
            player_data.get('passer_rating_allowed'),
            player_data.get('interception_rate'),
            player_data.get('tackle_efficiency'),
            player_data.get('missed_tackle_rate'),
            player_data.get('run_stop_percentage'),
            player_data.get('deep_ball_coverage'),
            player_data.get('blitz_effectiveness'),
            player_data.get('communication_leadership'),
            player_data.get('data_source', 'api')
        ))
        self.conn.commit()
    
    def get_safety_stats(self, team: Optional[str] = None, player_id: Optional[int] = None, 
                        week: Optional[int] = None, season: int = 2025) -> List[Dict]:
        """Get safety statistics."""
        query = "SELECT * FROM safety_stats WHERE season = ?"
        params = [season]
        
        if team:
            query += " AND team = ?"
            params.append(team)
        if player_id:
            query += " AND player_id = ?"
            params.append(player_id)
        if week:
            query += " AND week = ?"
            params.append(week)
        
        query += " ORDER BY tackles DESC"
        
        self.cursor.execute(query, params)
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    # Offensive Line Methods
    def save_offensive_line_stats(self, player_data: Dict[str, Any]) -> None:
        """Save offensive line player statistics."""
        self.cursor.execute("""
            INSERT OR REPLACE INTO offensive_line_stats 
            (player_id, player_name, team, position, week, season, games_played, games_started, 
             pass_block_win_rate, run_block_win_rate, pressure_rate_allowed, sack_rate_allowed, 
             hurry_rate_allowed, hit_rate_allowed, run_blocking_grade, pass_blocking_grade, 
             penalty_rate, snap_count, durability, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_data.get('player_id'),
            player_data.get('player_name'),
            player_data.get('team'),
            player_data.get('position'),
            player_data.get('week'),
            player_data.get('season'),
            player_data.get('games_played', 0),
            player_data.get('games_started', 0),
            player_data.get('pass_block_win_rate'),
            player_data.get('run_block_win_rate'),
            player_data.get('pressure_rate_allowed'),
            player_data.get('sack_rate_allowed'),
            player_data.get('hurry_rate_allowed'),
            player_data.get('hit_rate_allowed'),
            player_data.get('run_blocking_grade'),
            player_data.get('pass_blocking_grade'),
            player_data.get('penalty_rate'),
            player_data.get('snap_count'),
            player_data.get('durability'),
            player_data.get('data_source', 'api')
        ))
        self.conn.commit()
    
    def get_offensive_line_stats(self, team: Optional[str] = None, player_id: Optional[int] = None, 
                               week: Optional[int] = None, season: int = 2025) -> List[Dict]:
        """Get offensive line statistics."""
        query = "SELECT * FROM offensive_line_stats WHERE season = ?"
        params = [season]
        
        if team:
            query += " AND team = ?"
            params.append(team)
        if player_id:
            query += " AND player_id = ?"
            params.append(player_id)
        if week:
            query += " AND week = ?"
            params.append(week)
        
        query += " ORDER BY games_started DESC"
        
        self.cursor.execute(query, params)
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    # Tight End Methods
    def save_tight_end_stats(self, player_data: Dict[str, Any]) -> None:
        """Save tight end player statistics."""
        self.cursor.execute("""
            INSERT OR REPLACE INTO tight_end_stats 
            (player_id, player_name, team, position, week, season, receptions, receiving_yards, 
             touchdowns, targets, yards_per_reception, catch_rate, target_share, route_running_grade, 
             blocking_grade, red_zone_efficiency, third_down_conversion_rate, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_data.get('player_id'),
            player_data.get('player_name'),
            player_data.get('team'),
            player_data.get('position'),
            player_data.get('week'),
            player_data.get('season'),
            player_data.get('receptions', 0),
            player_data.get('receiving_yards', 0),
            player_data.get('touchdowns', 0),
            player_data.get('targets', 0),
            player_data.get('yards_per_reception', 0),
            player_data.get('catch_rate'),
            player_data.get('target_share'),
            player_data.get('route_running_grade'),
            player_data.get('blocking_grade'),
            player_data.get('red_zone_efficiency'),
            player_data.get('third_down_conversion_rate'),
            player_data.get('data_source', 'api')
        ))
        self.conn.commit()
    
    def get_tight_end_stats(self, team: Optional[str] = None, player_id: Optional[int] = None, 
                           week: Optional[int] = None, season: int = 2025) -> List[Dict]:
        """Get tight end statistics."""
        query = "SELECT * FROM tight_end_stats WHERE season = ?"
        params = [season]
        
        if team:
            query += " AND team = ?"
            params.append(team)
        if player_id:
            query += " AND player_id = ?"
            params.append(player_id)
        if week:
            query += " AND week = ?"
            params.append(week)
        
        query += " ORDER BY receiving_yards DESC"
        
        self.cursor.execute(query, params)
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    # Wide Receiver Methods
    def save_wide_receiver_stats(self, player_data: Dict[str, Any]) -> None:
        """Save wide receiver player statistics."""
        self.cursor.execute("""
            INSERT OR REPLACE INTO wide_receiver_stats 
            (player_id, player_name, team, position, week, season, receptions, receiving_yards, 
             touchdowns, targets, yards_per_reception, catch_rate, target_share, route_running_grade, 
             separation, contested_catch_rate, yards_after_catch, red_zone_efficiency, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_data.get('player_id'),
            player_data.get('player_name'),
            player_data.get('team'),
            player_data.get('position'),
            player_data.get('week'),
            player_data.get('season'),
            player_data.get('receptions', 0),
            player_data.get('receiving_yards', 0),
            player_data.get('touchdowns', 0),
            player_data.get('targets', 0),
            player_data.get('yards_per_reception', 0),
            player_data.get('catch_rate'),
            player_data.get('target_share'),
            player_data.get('route_running_grade'),
            player_data.get('separation'),
            player_data.get('contested_catch_rate'),
            player_data.get('yards_after_catch'),
            player_data.get('red_zone_efficiency'),
            player_data.get('data_source', 'api')
        ))
        self.conn.commit()
    
    def get_wide_receiver_stats(self, team: Optional[str] = None, player_id: Optional[int] = None, 
                               week: Optional[int] = None, season: int = 2025) -> List[Dict]:
        """Get wide receiver statistics."""
        query = "SELECT * FROM wide_receiver_stats WHERE season = ?"
        params = [season]
        
        if team:
            query += " AND team = ?"
            params.append(team)
        if player_id:
            query += " AND player_id = ?"
            params.append(player_id)
        if week:
            query += " AND week = ?"
            params.append(week)
        
        query += " ORDER BY receiving_yards DESC"
        
        self.cursor.execute(query, params)
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    # Running Back Methods
    def save_running_back_stats(self, player_data: Dict[str, Any]) -> None:
        """Save running back player statistics."""
        self.cursor.execute("""
            INSERT OR REPLACE INTO running_back_stats 
            (player_id, player_name, team, position, week, season, rushing_yards, rushing_attempts, 
             rushing_touchdowns, receptions, receiving_yards, receiving_touchdowns, fumbles, 
             yards_per_carry, rushing_grade, receiving_grade, pass_blocking_grade, fumble_rate, 
             breakaway_run_rate, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_data.get('player_id'),
            player_data.get('player_name'),
            player_data.get('team'),
            player_data.get('position'),
            player_data.get('week'),
            player_data.get('season'),
            player_data.get('rushing_yards', 0),
            player_data.get('rushing_attempts', 0),
            player_data.get('rushing_touchdowns', 0),
            player_data.get('receptions', 0),
            player_data.get('receiving_yards', 0),
            player_data.get('receiving_touchdowns', 0),
            player_data.get('fumbles', 0),
            player_data.get('yards_per_carry'),
            player_data.get('rushing_grade'),
            player_data.get('receiving_grade'),
            player_data.get('pass_blocking_grade'),
            player_data.get('fumble_rate'),
            player_data.get('breakaway_run_rate'),
            player_data.get('data_source', 'api')
        ))
        self.conn.commit()
    
    def get_running_back_stats(self, team: Optional[str] = None, player_id: Optional[int] = None, 
                              week: Optional[int] = None, season: int = 2025) -> List[Dict]:
        """Get running back statistics."""
        query = "SELECT * FROM running_back_stats WHERE season = ?"
        params = [season]
        
        if team:
            query += " AND team = ?"
            params.append(team)
        if player_id:
            query += " AND player_id = ?"
            params.append(player_id)
        if week:
            query += " AND week = ?"
            params.append(week)
        
        query += " ORDER BY rushing_yards DESC"
        
        self.cursor.execute(query, params)
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    # Quarterback Methods
    def save_quarterback_stats(self, player_data: Dict[str, Any]) -> None:
        """Save quarterback player statistics."""
        self.cursor.execute("""
            INSERT OR REPLACE INTO quarterback_stats 
            (player_id, player_name, team, position, week, season, passing_yards, passing_touchdowns, 
             interceptions, completions, attempts, passer_rating, qbr, epa, cpoe, pocket_presence_grade, 
             deep_ball_accuracy, red_zone_efficiency, mobility_rushing_yards, mobility_rushing_tds, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_data.get('player_id'),
            player_data.get('player_name'),
            player_data.get('team'),
            player_data.get('position'),
            player_data.get('week'),
            player_data.get('season'),
            player_data.get('passing_yards', 0),
            player_data.get('passing_touchdowns', 0),
            player_data.get('interceptions', 0),
            player_data.get('completions', 0),
            player_data.get('attempts', 0),
            player_data.get('passer_rating', 0),
            player_data.get('qbr'),
            player_data.get('epa'),
            player_data.get('cpoe'),
            player_data.get('pocket_presence_grade'),
            player_data.get('deep_ball_accuracy'),
            player_data.get('red_zone_efficiency'),
            player_data.get('mobility_rushing_yards'),
            player_data.get('mobility_rushing_tds'),
            player_data.get('data_source', 'api')
        ))
        self.conn.commit()
    
    def get_quarterback_stats(self, team: Optional[str] = None, player_id: Optional[int] = None, 
                             week: Optional[int] = None, season: int = 2025) -> List[Dict]:
        """Get quarterback statistics."""
        query = "SELECT * FROM quarterback_stats WHERE season = ?"
        params = [season]
        
        if team:
            query += " AND team = ?"
            params.append(team)
        if player_id:
            query += " AND player_id = ?"
            params.append(player_id)
        if week:
            query += " AND week = ?"
            params.append(week)
        
        query += " ORDER BY passer_rating DESC"
        
        self.cursor.execute(query, params)
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    # Special Teams Methods
    def save_special_teams_stats(self, player_data: Dict[str, Any]) -> None:
        """Save special teams player statistics."""
        self.cursor.execute("""
            INSERT OR REPLACE INTO special_teams_stats 
            (player_id, player_name, team, position, week, season, field_goals_made, field_goals_attempted, 
             extra_points_made, extra_points_attempted, punts, punt_yards, kick_return_avg, punt_return_avg, 
             touchback_rate, inside_20_percentage, hang_time, return_touchdowns, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_data.get('player_id'),
            player_data.get('player_name'),
            player_data.get('team'),
            player_data.get('position'),
            player_data.get('week'),
            player_data.get('season'),
            player_data.get('field_goals_made', 0),
            player_data.get('field_goals_attempted', 0),
            player_data.get('extra_points_made', 0),
            player_data.get('extra_points_attempted', 0),
            player_data.get('punts', 0),
            player_data.get('punt_yards', 0),
            player_data.get('kick_return_avg'),
            player_data.get('punt_return_avg'),
            player_data.get('touchback_rate'),
            player_data.get('inside_20_percentage'),
            player_data.get('hang_time'),
            player_data.get('return_touchdowns'),
            player_data.get('data_source', 'api')
        ))
        self.conn.commit()
    
    def get_special_teams_stats(self, team: Optional[str] = None, player_id: Optional[int] = None, 
                               week: Optional[int] = None, season: int = 2025) -> List[Dict]:
        """Get special teams statistics."""
        query = "SELECT * FROM special_teams_stats WHERE season = ?"
        params = [season]
        
        if team:
            query += " AND team = ?"
            params.append(team)
        if player_id:
            query += " AND player_id = ?"
            params.append(player_id)
        if week:
            query += " AND week = ?"
            params.append(week)
        
        query += " ORDER BY field_goals_made DESC"
        
        self.cursor.execute(query, params)
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    # Team Metadata Methods
    def save_team_metadata(self, team_data: Dict[str, Any]) -> None:
        """Save team metadata."""
        self.cursor.execute("""
            INSERT OR REPLACE INTO team_metadata 
            (team_name, team_abbreviation, division, conference, last_api_update, data_source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            team_data.get('team_name'),
            team_data.get('team_abbreviation'),
            team_data.get('division'),
            team_data.get('conference'),
            team_data.get('last_api_update'),
            team_data.get('data_source', 'api')
        ))
        self.conn.commit()
    
    def get_team_metadata(self, team_abbreviation: Optional[str] = None) -> List[Dict]:
        """Get team metadata."""
        if team_abbreviation:
            self.cursor.execute("SELECT * FROM team_metadata WHERE team_abbreviation = ?", (team_abbreviation,))
        else:
            self.cursor.execute("SELECT * FROM team_metadata ORDER BY team_name")
        
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    # Utility Methods
    def get_all_teams(self) -> List[str]:
        """Get list of all teams."""
        self.cursor.execute("SELECT DISTINCT team FROM (SELECT team FROM defensive_line_stats UNION SELECT team FROM linebacker_stats UNION SELECT team FROM cornerback_stats UNION SELECT team FROM safety_stats UNION SELECT team FROM offensive_line_stats UNION SELECT team FROM tight_end_stats UNION SELECT team FROM wide_receiver_stats UNION SELECT team FROM running_back_stats UNION SELECT team FROM quarterback_stats UNION SELECT team FROM special_teams_stats) ORDER BY team")
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_players_by_position(self, position: str, team: Optional[str] = None, 
                               week: Optional[int] = None, season: int = 2025) -> List[Dict]:
        """Get all players for a specific position."""
        position_tables = {
            'DL': 'defensive_line_stats',
            'LB': 'linebacker_stats', 
            'CB': 'cornerback_stats',
            'S': 'safety_stats',
            'OL': 'offensive_line_stats',
            'TE': 'tight_end_stats',
            'WR': 'wide_receiver_stats',
            'RB': 'running_back_stats',
            'QB': 'quarterback_stats',
            'ST': 'special_teams_stats'
        }
        
        if position not in position_tables:
            return []
        
        table = position_tables[position]
        query = f"SELECT * FROM {table} WHERE season = ?"
        params = [season]
        
        if team:
            query += " AND team = ?"
            params.append(team)
        if week:
            query += " AND week = ?"
            params.append(week)
        
        query += " ORDER BY CASE WHEN data_source = 'starting_qb_stats' THEN 1 WHEN data_source = 'current_roster_stats' THEN 2 WHEN data_source = 'api' THEN 3 WHEN data_source = 'realistic_roster' THEN 4 WHEN data_source = 'sample_data' THEN 5 ELSE 6 END, player_name"
        
        self.cursor.execute(query, params)
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def get_team_stats(self, team_name: str, week: Optional[int] = None, season: int = 2025) -> Dict[str, List[Dict]]:
        """Get all position group stats for a team."""
        positions = ['DL', 'LB', 'CB', 'S', 'OL', 'TE', 'WR', 'RB', 'QB', 'ST']
        team_stats = {}
        
        for position in positions:
            team_stats[position] = self.get_players_by_position(position, team_name, week, season)
        
        return team_stats
    
    def clear_week_data(self, week: int, season: int = 2025) -> None:
        """Clear all data for a specific week and season."""
        tables = [
            'defensive_line_stats', 'linebacker_stats', 'cornerback_stats', 'safety_stats',
            'offensive_line_stats', 'tight_end_stats', 'wide_receiver_stats', 
            'running_back_stats', 'quarterback_stats', 'special_teams_stats'
        ]
        
        for table in tables:
            self.cursor.execute(f"DELETE FROM {table} WHERE week = ? AND season = ?", (week, season))
        
        self.conn.commit()
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get statistics about the database."""
        tables = [
            'defensive_line_stats', 'linebacker_stats', 'cornerback_stats', 'safety_stats',
            'offensive_line_stats', 'tight_end_stats', 'wide_receiver_stats', 
            'running_back_stats', 'quarterback_stats', 'special_teams_stats'
        ]
        
        stats = {}
        for table in tables:
            self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = self.cursor.fetchone()[0]
        
        return stats
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
