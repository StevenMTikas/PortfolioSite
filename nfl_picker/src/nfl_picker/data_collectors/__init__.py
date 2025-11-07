"""
Data collectors package for NFL statistics.
Handles API integration and web scraping for player statistics.
"""

from .api_collector import APIDataCollector
from .web_scraper import WebScraper

__all__ = ['APIDataCollector', 'WebScraper']
