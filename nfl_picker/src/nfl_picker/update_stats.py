#!/usr/bin/env python3
"""
Weekly statistics update script for NFL position data.
Runs every Tuesday to update player statistics from API and web scraping.
"""

import argparse
import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nfl_picker.stats_database import NFLStatsDatabase
from nfl_picker.data_collectors.api_collector import APIDataCollector
from nfl_picker.data_collectors.web_scraper import WebScraper
from nfl_picker.enhanced_web_scraper import EnhancedWebScraper
from nfl_picker.utils import get_current_nfl_week


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/stats_update.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def validate_environment() -> bool:
    """Validate that required environment variables are set."""
    required_vars = ['SPORTSDATA_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    return True


def update_api_data(week: int, season: int, logger: logging.Logger) -> Dict[str, int]:
    """
    Update database with API data for all positions.
    
    Args:
        week: Week number
        season: Season year
        logger: Logger instance
        
    Returns:
        Dictionary with position update counts
    """
    logger.info(f"Starting API data update for week {week}, season {season}")
    
    try:
        api_collector = APIDataCollector()
        results = api_collector.update_all_positions(week, season)
        api_collector.close()
        
        logger.info(f"API update completed: {results}")
        return results
        
    except Exception as e:
        logger.error(f"API update failed: {e}")
        return {}


def update_scraped_data(week: int, season: int, logger: logging.Logger) -> Dict[str, int]:
    """
    Update database with scraped data for advanced metrics.
    
    Args:
        week: Week number
        season: Season year
        logger: Logger instance
        
    Returns:
        Dictionary with position update counts
    """
    logger.info(f"Starting enhanced web scraping for week {week}, season {season}")
    
    try:
        scraper = EnhancedWebScraper()
        
        # Test scraping sources
        logger.info("Testing web scraping sources...")
        
        # Scrape all positions with enhanced scraper
        results = scraper.update_all_positions_with_scraping(week, season)
        
        scraper.close()
        logger.info(f"Enhanced web scraping completed: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Enhanced web scraping failed: {e}")
        return {}


def validate_data_completeness(week: int, season: int, logger: logging.Logger) -> Dict[str, Any]:
    """
    Validate that data was updated correctly.
    
    Args:
        week: Week number
        season: Season year
        logger: Logger instance
        
    Returns:
        Validation results dictionary
    """
    logger.info(f"Validating data completeness for week {week}, season {season}")
    
    try:
        stats_db = NFLStatsDatabase()
        
        # Get database statistics
        db_stats = stats_db.get_database_stats()
        
        # Check for recent updates
        validation_results = {
            'database_stats': db_stats,
            'week': week,
            'season': season,
            'validation_time': datetime.now().isoformat()
        }
        
        # Check each position table for recent data
        positions = ['defensive_line_stats', 'linebacker_stats', 'cornerback_stats', 
                    'safety_stats', 'offensive_line_stats', 'tight_end_stats', 
                    'wide_receiver_stats', 'running_back_stats', 'quarterback_stats', 
                    'special_teams_stats']
        
        recent_data_counts = {}
        for position in positions:
            try:
                # Get count of records for this week
                table_name = position
                query = f"SELECT COUNT(*) FROM {table_name} WHERE week = ? AND season = ?"
                stats_db.cursor.execute(query, (week, season))
                count = stats_db.cursor.fetchone()[0]
                recent_data_counts[position] = count
            except Exception as e:
                logger.error(f"Error checking {position}: {e}")
                recent_data_counts[position] = 0
        
        validation_results['recent_data_counts'] = recent_data_counts
        
        # Calculate completeness score
        total_positions = len(positions)
        positions_with_data = sum(1 for count in recent_data_counts.values() if count > 0)
        completeness_score = (positions_with_data / total_positions) * 100
        
        validation_results['completeness_score'] = completeness_score
        validation_results['positions_with_data'] = positions_with_data
        validation_results['total_positions'] = total_positions
        
        stats_db.close()
        
        logger.info(f"Data validation completed. Completeness: {completeness_score:.1f}%")
        return validation_results
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return {'error': str(e)}


def clear_old_data(week: int, season: int, logger: logging.Logger) -> None:
    """
    Clear old data for the specified week before updating.
    
    Args:
        week: Week number
        season: Season year
        logger: Logger instance
    """
    logger.info(f"Clearing old data for week {week}, season {season}")
    
    try:
        stats_db = NFLStatsDatabase()
        stats_db.clear_week_data(week, season)
        stats_db.close()
        logger.info("Old data cleared successfully")
    except Exception as e:
        logger.error(f"Failed to clear old data: {e}")


def main():
    """Main update script function."""
    parser = argparse.ArgumentParser(description='Update NFL player statistics')
    parser.add_argument('--week', type=int, help='Week number (default: current week)')
    parser.add_argument('--season', type=int, default=2025, help='Season year (default: 2025)')
    parser.add_argument('--api-only', action='store_true', help='Only update API data')
    parser.add_argument('--scrape-only', action='store_true', help='Only update scraped data')
    parser.add_argument('--clear-first', action='store_true', help='Clear existing data first')
    parser.add_argument('--validate', action='store_true', help='Only validate existing data')
    
    args = parser.parse_args()
    
    # Set up logging
    os.makedirs('logs', exist_ok=True)
    logger = setup_logging()
    
    # Determine week
    if args.week:
        week = args.week
    else:
        week = get_current_nfl_week()
    
    season = args.season
    
    logger.info(f"Starting NFL stats update for week {week}, season {season}")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        sys.exit(1)
    
    # Clear old data if requested
    if args.clear_first:
        clear_old_data(week, season, logger)
    
    # Update API data
    api_results = {}
    if not args.scrape_only:
        api_results = update_api_data(week, season, logger)
    
    # Update scraped data
    scrape_results = {}
    if not args.api_only:
        scrape_results = update_scraped_data(week, season, logger)
    
    # Validate data completeness
    validation_results = validate_data_completeness(week, season, logger)
    
    # Summary
    logger.info("=== UPDATE SUMMARY ===")
    logger.info(f"Week: {week}, Season: {season}")
    logger.info(f"API Results: {api_results}")
    logger.info(f"Scrape Results: {scrape_results}")
    logger.info(f"Validation: {validation_results}")
    
    # Check if validation passed
    if 'completeness_score' in validation_results:
        completeness = validation_results['completeness_score']
        if completeness >= 80:
            logger.info(f"✅ Update successful! Completeness: {completeness:.1f}%")
            sys.exit(0)
        else:
            logger.warning(f"⚠️ Update completed with low completeness: {completeness:.1f}%")
            sys.exit(1)
    else:
        logger.error("❌ Update failed - validation error")
        sys.exit(1)


if __name__ == "__main__":
    main()
