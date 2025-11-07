"""
Utility functions for NFL Picker system.
Contains shared helper functions used across the application.
"""

from datetime import datetime
from typing import Optional, Dict, Any
import time
from functools import wraps


def get_current_nfl_week() -> int:
    """
    Calculate current NFL week based on date.
    
    Returns:
        int: Current NFL week (1-18)
    """
    today = datetime.now()
    # NFL season typically starts first week of September
    september_1 = datetime(today.year if today.month >= 9 else today.year - 1, 9, 1)
    days_since_sept_1 = (today - september_1).days
    week = (days_since_sept_1 // 7) + 1
    
    # NFL regular season is 18 weeks
    week = min(max(week, 1), 18)
    
    return week


def create_analysis_inputs(
    team1: Optional[str] = None,
    team2: Optional[str] = None,
    home_team: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create standardized inputs for NFL analysis.
    
    Args:
        team1: First team name (optional)
        team2: Second team name (optional)
        home_team: Home team name (optional)
        **kwargs: Additional input parameters
    
    Returns:
        Dict containing analysis inputs
    """
    inputs = {
        'current_year': '2025',
        'nfl_season': '2025 NFL Season',
        'current_week': f"Week {get_current_nfl_week()}",
        'injury_analysis': 'Current week injury reports and performance impact analysis'
    }
    
    # Add team-specific inputs if provided
    if team1 and team2:
        away_team = team1 if team2 == home_team else team2
        inputs.update({
            'team1': team1,
            'team2': team2,
            'home_team': home_team or team2,
            'away_team': away_team,
        })
    
    # Add any additional inputs
    inputs.update(kwargs)
    
    return inputs


def is_network_error(exception: Exception) -> bool:
    """
    Check if an exception is a network-related error.
    
    Args:
        exception: The exception to check
    
    Returns:
        bool: True if it's a network error, False otherwise
    """
    error_msg = str(exception).lower()
    network_keywords = [
        'ssl', 'connection', 'network', 'timeout', 'socket',
        'recv', 'read', 'httpcore', 'httpx', 'openai', 'sslobj'
    ]
    return any(keyword in error_msg for keyword in network_keywords)


def retry_on_network_error(max_retries: int = 3, initial_delay: int = 5):
    """
    Decorator to retry a function on network errors with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds between retries
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    if is_network_error(e) and attempt < max_retries - 1:
                        print(f"Network error detected (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}...")
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                        continue
                    elif attempt == max_retries - 1 and is_network_error(e):
                        raise Exception(
                            f"Network connectivity issue after {max_retries} attempts. "
                            f"This appears to be an SSL/network connectivity issue. Try:\n"
                            f"1. Check your internet connection\n"
                            f"2. Use a VPN if on corporate network\n"
                            f"3. Restart your network adapter\n"
                            f"4. Try again later"
                        )
                    else:
                        # Non-network error or final attempt, re-raise
                        raise
            
        return wrapper
    return decorator

