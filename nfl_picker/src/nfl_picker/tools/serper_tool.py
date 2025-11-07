from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import requests
from datetime import datetime
from nfl_picker.config import CONFIG

class SerperToolInput(BaseModel):
    """Input schema for SerperTool."""
    query: str = Field(None, description="The search query to look up on the web")
    description: str = Field(None, description="Alternative search query field")
    metadata: dict = Field(None, description="Additional metadata")

class SerperTool(BaseTool):
    name: str = "serper_search"
    description: str = (
        "Search the web for current NFL information using Serper API. Use this to find up-to-date statistics, "
        "injury reports, and 2025 NFL data. The tool automatically adds current date/time to searches to ensure "
        "results are recent and relevant. Always specify 2025 in your search query to ensure current year data."
    )
    args_schema: Type[BaseModel] = SerperToolInput

    def _run(self, query: str = None, description: str = None, metadata: dict = None, **kwargs) -> str:
        """Search the web for information using Serper API."""
        try:
            # Handle different input formats from CrewAI
            search_query = None
            
            # Try to get query from various sources
            if query:
                search_query = query
            elif description:
                search_query = description
            elif 'query' in kwargs:
                search_query = kwargs['query']
            elif 'description' in kwargs:
                search_query = kwargs['description']
            else:
                return "Error: No search query provided"
            
            # Ensure query is a string
            if not isinstance(search_query, str):
                search_query = str(search_query)

            # Add current date/time to search query for recency
            current_datetime = datetime.now()
            current_date_str = current_datetime.strftime("%B %d, %Y")  # e.g., "January 15, 2025"

            # Enhance query with timestamp for current data
            # Only add if not already present to avoid duplication
            if "2025" not in search_query and current_datetime.year == 2025:
                search_query = f"{search_query} 2025"

            # Add "as of [date]" or "latest" to emphasize recency
            if "latest" not in search_query.lower() and "recent" not in search_query.lower():
                search_query = f"{search_query} latest as of {current_date_str}"

            # Get API key from configuration
            api_key = CONFIG['serper_api_key']
            if not api_key or api_key == 'your_serper_api_key_here':
                return "Error: SERPER_API_KEY not configured. Please set your Serper API key in the .env file."

            # Serper API endpoint
            url = "https://google.serper.dev/search"

            headers = {
                'X-API-KEY': api_key,
                'Content-Type': 'application/json'
            }

            # Add date range filter to get recent results (last 30 days)
            payload = {
                'q': search_query,
                'num': 10,  # Number of results to return
                'tbs': 'qdr:m'  # Filter to results from the past month for recency
            }
            
            response = requests.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                data = response.json()

                # Format the results
                results = []
                if 'organic' in data:
                    for item in data['organic'][:5]:  # Top 5 results
                        title = item.get('title', 'No title')
                        snippet = item.get('snippet', 'No description')
                        link = item.get('link', 'No link')
                        date = item.get('date', '')  # Include publication date if available

                        result_text = f"Title: {title}\nDescription: {snippet}\nLink: {link}"
                        if date:
                            result_text += f"\nPublished: {date}"
                        results.append(result_text + "\n")

                if results:
                    header = f"Search results for '{search_query}' (searched on {current_date_str}):\n\n"
                    return header + "\n".join(results)
                else:
                    return f"No results found for '{search_query}' (searched on {current_date_str})"
            else:
                return f"Error: Serper API returned status code {response.status_code}"

        except Exception as e:
            return f"Error performing web search: {str(e)}"
