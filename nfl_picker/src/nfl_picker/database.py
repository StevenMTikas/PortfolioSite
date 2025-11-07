"""
Database manager for NFL Picker system.
Handles all database operations for predictions, results, and accuracy tracking.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime


class NFLDatabase:
    """Database manager for NFL predictions and results."""
    
    def __init__(self, db_path: str = "data/nfl_learning.db"):
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
        """Create database tables if they don't exist."""
        # Predictions table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT UNIQUE,
                team1 TEXT NOT NULL,
                team2 TEXT NOT NULL,
                home_team TEXT NOT NULL,
                predicted_winner TEXT NOT NULL,
                predicted_score_home INTEGER,
                predicted_score_away INTEGER,
                confidence_level REAL,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                analysis_data TEXT,
                week INTEGER,
                season INTEGER
            )
        ''')
        
        # Game results table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT UNIQUE,
                team1 TEXT NOT NULL,
                team2 TEXT NOT NULL,
                home_team TEXT NOT NULL,
                actual_winner TEXT,
                actual_score_home INTEGER,
                actual_score_away INTEGER,
                game_date DATE,
                weather_conditions TEXT,
                attendance INTEGER,
                result_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                week INTEGER,
                season INTEGER
            )
        ''')
        
        # Prediction accuracy table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                game_result_id INTEGER,
                was_correct BOOLEAN,
                score_difference INTEGER,
                confidence_vs_accuracy REAL,
                FOREIGN KEY (prediction_id) REFERENCES predictions (id),
                FOREIGN KEY (game_result_id) REFERENCES game_results (id)
            )
        ''')
        
        self.conn.commit()
    
    def save_prediction(
        self,
        game_id: str,
        team1: str,
        team2: str,
        home_team: str,
        predicted_winner: str,
        predicted_score_home: int,
        predicted_score_away: int,
        confidence_level: float,
        analysis_data: Dict[str, Any],
        week: int,
        season: int = 2025
    ) -> None:
        """
        Save a game prediction to the database.
        
        Args:
            game_id: Unique identifier for the game
            team1: First team name
            team2: Second team name
            home_team: Home team name
            predicted_winner: Predicted winning team
            predicted_score_home: Predicted home team score
            predicted_score_away: Predicted away team score
            confidence_level: Confidence level (0.0-1.0)
            analysis_data: Full analysis data as dictionary
            week: NFL week number
            season: NFL season year
        """
        self.cursor.execute("""
            INSERT OR REPLACE INTO predictions 
            (game_id, team1, team2, home_team, predicted_winner, predicted_score_home, 
             predicted_score_away, confidence_level, analysis_data, week, season)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id, team1, team2, home_team, predicted_winner,
            predicted_score_home, predicted_score_away, confidence_level,
            json.dumps(analysis_data), week, season
        ))
        self.conn.commit()
    
    def get_predictions(self, limit: Optional[int] = None) -> List[Tuple]:
        """
        Get all predictions ordered by date.
        
        Args:
            limit: Maximum number of predictions to return (None for all)
        
        Returns:
            List of prediction tuples
        """
        query = """
            SELECT game_id, team1, team2, predicted_winner, prediction_date 
            FROM predictions 
            ORDER BY prediction_date DESC
        """
        if limit:
            query += f" LIMIT {limit}"
        
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    def get_team_from_prediction(self, game_id: str, team_field: str) -> str:
        """
        Get team name from prediction record.
        
        Args:
            game_id: Game identifier
            team_field: Field name ('team1', 'team2', or 'home_team')
        
        Returns:
            Team name
        """
        self.cursor.execute(
            f"SELECT {team_field} FROM predictions WHERE game_id = ?",
            (game_id,)
        )
        result = self.cursor.fetchone()
        return result[0] if result else None
    
    def save_game_result(
        self,
        game_id: str,
        team1: str,
        team2: str,
        home_team: str,
        actual_winner: str,
        actual_score_home: int,
        actual_score_away: int,
        game_date: Optional[str] = None,
        weather_conditions: Optional[str] = None,
        week: int = 1,
        season: int = 2025
    ) -> None:
        """
        Save actual game result to the database.
        
        Args:
            game_id: Unique identifier for the game
            team1: First team name
            team2: Second team name
            home_team: Home team name
            actual_winner: Actual winning team
            actual_score_home: Actual home team score
            actual_score_away: Actual away team score
            game_date: Date the game was played
            weather_conditions: Weather conditions during the game
            week: NFL week number
            season: NFL season year
        """
        self.cursor.execute("""
            INSERT OR REPLACE INTO game_results 
            (game_id, team1, team2, home_team, actual_winner, actual_score_home, 
             actual_score_away, game_date, weather_conditions, week, season)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id, team1, team2, home_team, actual_winner,
            actual_score_home, actual_score_away, game_date,
            weather_conditions, week, season
        ))
        self.conn.commit()
    
    def calculate_and_store_accuracy(self, game_id: str) -> None:
        """
        Calculate and store prediction accuracy for a game.
        
        Args:
            game_id: Game identifier
        """
        self.cursor.execute("""
            SELECT p.id, p.predicted_winner, p.confidence_level, 
                   gr.actual_winner, gr.actual_score_home, gr.actual_score_away
            FROM predictions p
            JOIN game_results gr ON p.game_id = gr.game_id
            WHERE p.game_id = ?
        """, (game_id,))
        
        result = self.cursor.fetchone()
        if not result:
            return
        
        pred_id, predicted_winner, confidence, actual_winner, home_score, away_score = result
        
        # Calculate accuracy metrics
        was_correct = predicted_winner == actual_winner
        score_difference = abs(home_score - away_score)
        confidence_vs_accuracy = confidence if was_correct else (1.0 - confidence)
        
        # Store accuracy record
        self.cursor.execute("""
            INSERT OR REPLACE INTO prediction_accuracy
            (prediction_id, game_result_id, was_correct, score_difference, confidence_vs_accuracy)
            VALUES (?, (SELECT id FROM game_results WHERE game_id = ?), ?, ?, ?)
        """, (pred_id, game_id, was_correct, score_difference, confidence_vs_accuracy))
        
        self.conn.commit()
    
    def get_accuracy_stats(self) -> Optional[Tuple]:
        """
        Get overall prediction accuracy statistics.
        
        Returns:
            Tuple of (total_predictions, correct_predictions, avg_confidence_accuracy, avg_score_difference)
        """
        self.cursor.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
                AVG(confidence_vs_accuracy) as avg_confidence_accuracy,
                AVG(score_difference) as avg_score_difference
            FROM prediction_accuracy
        """)
        return self.cursor.fetchone()
    
    def get_recent_games_with_results(self, limit: int = 10) -> List[Tuple]:
        """
        Get recent predictions with their actual results.
        
        Args:
            limit: Maximum number of games to return
        
        Returns:
            List of game result tuples
        """
        self.cursor.execute("""
            SELECT p.team1, p.team2, p.predicted_winner, p.confidence_level,
                   gr.actual_winner, gr.actual_score_home, gr.actual_score_away,
                   pa.was_correct, pa.score_difference
            FROM predictions p
            JOIN game_results gr ON p.game_id = gr.game_id
            JOIN prediction_accuracy pa ON p.id = pa.prediction_id
            ORDER BY p.prediction_date DESC
            LIMIT ?
        """, (limit,))
        return self.cursor.fetchall()
    
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

