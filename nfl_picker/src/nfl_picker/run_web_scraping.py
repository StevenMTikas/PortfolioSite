#!/usr/bin/env python3
"""
Web scraping script to populate NFL statistics database with real data.
Scrapes from ESPN, NFL.com, and other sources to get actual player statistics.
"""

import argparse
import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nfl_picker.enhanced_web_scraper import EnhancedWebScraper
from nfl_picker.stats_database import NFLStatsDatabase
from nfl_picker.utils import get_current_nfl_week


def setup_logging():
    """Set up logging configuration."""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/web_scraping.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def test_scraping_sources():
    """Test which scraping sources are available."""
    logger = logging.getLogger(__name__)
    scraper = EnhancedWebScraper()
    
    logger.info("Testing web scraping sources...")
    
    # Test ESPN
    try:
        espn_data = scraper.scrape_espn_player_stats('QB')
        logger.info(f"ESPN: Found {len(espn_data)} QB players")
    except Exception as e:
        logger.error(f"ESPN scraping failed: {e}")
    
    # Test NFL.com
    try:
        nfl_data = scraper.scrape_nfl_com_stats('QB')
        logger.info(f"NFL.com: Found {len(nfl_data)} QB players")
    except Exception as e:
        logger.error(f"NFL.com scraping failed: {e}")
    
    scraper.close()


def scrape_position_data(position: str, week: int, season: int = 2025) -> int:
    """Scrape data for a specific position."""
    logger = logging.getLogger(__name__)
    scraper = EnhancedWebScraper()
    
    try:
        logger.info(f"Scraping {position} data for week {week}")
        count = scraper.update_database_with_scraped_data(position, week, season)
        logger.info(f"Updated {count} {position} records")
        return count
    except Exception as e:
        logger.error(f"Error scraping {position}: {e}")
        return 0
    finally:
        scraper.close()


def scrape_all_positions(week: int, season: int = 2025) -> Dict[str, int]:
    """Scrape data for all positions."""
    logger = logging.getLogger(__name__)
    scraper = EnhancedWebScraper()
    
    try:
        logger.info(f"Starting comprehensive web scraping for week {week}, season {season}")
        results = scraper.update_all_positions_with_scraping(week, season)
        
        total_updated = sum(results.values())
        logger.info(f"Web scraping completed. Total records updated: {total_updated}")
        
        return results
    except Exception as e:
        logger.error(f"Error in comprehensive scraping: {e}")
        return {}
    finally:
        scraper.close()


def validate_scraped_data(week: int, season: int = 2025) -> Dict[str, Any]:
    """Validate that scraped data was saved correctly."""
    logger = logging.getLogger(__name__)
    
    try:
        db = NFLStatsDatabase()
        
        # Check each position table
        positions = ['QB', 'RB', 'WR', 'TE', 'DL', 'LB', 'CB', 'S']
        validation_results = {}
        
        for position in positions:
            try:
                if position == 'QB':
                    stats = db.get_quarterback_stats(week=week, season=season)
                elif position == 'RB':
                    stats = db.get_running_back_stats(week=week, season=season)
                elif position == 'WR':
                    stats = db.get_wide_receiver_stats(week=week, season=season)
                elif position == 'TE':
                    stats = db.get_tight_end_stats(week=week, season=season)
                elif position == 'DL':
                    stats = db.get_defensive_line_stats(week=week, season=season)
                elif position == 'LB':
                    stats = db.get_linebacker_stats(week=week, season=season)
                elif position == 'CB':
                    stats = db.get_cornerback_stats(week=week, season=season)
                elif position == 'S':
                    stats = db.get_safety_stats(week=week, season=season)
                
                # Count records with actual data (non-zero stats)
                records_with_data = 0
                for stat in stats:
                    # Check if any meaningful stat is > 0
                    stat_fields = ['passing_yards', 'rushing_yards', 'receiving_yards', 'tackles', 'sacks']
                    if any(stat.get(field, 0) > 0 for field in stat_fields):
                        records_with_data += 1
                
                validation_results[position] = {
                    'total_records': len(stats),
                    'records_with_data': records_with_data,
                    'data_percentage': (records_with_data / len(stats) * 100) if stats else 0
                }
                
            except Exception as e:
                logger.error(f"Error validating {position}: {e}")
                validation_results[position] = {'error': str(e)}
        
        db.close()
        return validation_results
        
    except Exception as e:
        logger.error(f"Error validating scraped data: {e}")
        return {'error': str(e)}


def create_sample_data(week: int, season: int = 2025) -> None:
    """Create sample data for testing when scraping fails."""
    logger = logging.getLogger(__name__)
    
    try:
        db = NFLStatsDatabase()
        
        # Sample QB data
        sample_qb = {
            'player_id': 99999,
            'player_name': 'Sample QB',
            'team': 'Kansas City Chiefs',
            'position': 'QB',
            'week': week,
            'season': season,
            'passing_yards': 2500,
            'passing_touchdowns': 20,
            'interceptions': 5,
            'completions': 200,
            'attempts': 300,
            'passer_rating': 95.5,
            'data_source': 'sample_data'
        }
        
        db.save_quarterback_stats(sample_qb)
        logger.info("Created sample QB data")
        
        # Sample RB data
        sample_rb = {
            'player_id': 99998,
            'player_name': 'Sample RB',
            'team': 'Kansas City Chiefs',
            'position': 'RB',
            'week': week,
            'season': season,
            'rushing_yards': 800,
            'rushing_attempts': 150,
            'rushing_touchdowns': 8,
            'receptions': 30,
            'receiving_yards': 200,
            'data_source': 'sample_data'
        }
        
        db.save_running_back_stats(sample_rb)
        logger.info("Created sample RB data")
        
        db.close()
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")


def main():
    """Main scraping function."""
    parser = argparse.ArgumentParser(description='Web scrape NFL statistics')
    parser.add_argument('--week', type=int, help='Week number (default: current week)')
    parser.add_argument('--season', type=int, default=2025, help='Season year (default: 2025)')
    parser.add_argument('--position', type=str, help='Specific position to scrape (QB, RB, WR, etc.)')
    parser.add_argument('--test-sources', action='store_true', help='Test scraping sources')
    parser.add_argument('--create-sample', action='store_true', help='Create sample data for testing')
    parser.add_argument('--validate', action='store_true', help='Validate scraped data')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    # Determine week
    if args.week:
        week = args.week
    else:
        week = get_current_nfl_week()
    
    season = args.season
    
    logger.info(f"Starting web scraping for week {week}, season {season}")
    
    # Test sources if requested
    if args.test_sources:
        test_scraping_sources()
        return
    
    # Create sample data if requested
    if args.create_sample:
        create_sample_data(week, season)
        logger.info("Sample data created successfully")
        return
    
    # Validate data if requested
    if args.validate:
        validation_results = validate_scraped_data(week, season)
        logger.info("Validation results:")
        for position, results in validation_results.items():
            if 'error' in results:
                logger.error(f"{position}: {results['error']}")
            else:
                logger.info(f"{position}: {results['records_with_data']}/{results['total_records']} records with data ({results['data_percentage']:.1f}%)")
        return
    
    # Scrape specific position
    if args.position:
        count = scrape_position_data(args.position, week, season)
        logger.info(f"Scraped {count} {args.position} records")
        return
    
    # Scrape all positions
    results = scrape_all_positions(week, season)
    
    # Show results
    logger.info("=== SCRAPING RESULTS ===")
    total_records = 0
    for position, count in results.items():
        logger.info(f"{position}: {count} records updated")
        total_records += count
    
    logger.info(f"Total records updated: {total_records}")
    
    # Validate results
    logger.info("Validating scraped data...")
    validation_results = validate_scraped_data(week, season)
    
    for position, results in validation_results.items():
        if 'error' not in results:
            data_pct = results['data_percentage']
            if data_pct > 0:
                logger.info(f"✅ {position}: {results['records_with_data']} records with data ({data_pct:.1f}%)")
            else:
                logger.warning(f"⚠️ {position}: No data found - may need different scraping approach")


if __name__ == "__main__":
    main()
