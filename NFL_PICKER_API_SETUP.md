# NFL Picker API Setup

This API server provides backend functionality for the NFL Picker web GUI.

## Prerequisites

- Python 3.8+
- All dependencies from the main project
- OpenAI API key (in `.env` file)
- SERPER_API_KEY (in `.env` file) - for web search functionality

## Installation

The API uses the same dependencies as the main NFL Picker project. Make sure you have installed all requirements.

## Running the API Server

```bash
python nfl_picker_api.py
```

The server will start on `http://localhost:8001` by default.

You can change the port by setting the `NFL_PICKER_PORT` environment variable:

```bash
NFL_PICKER_PORT=8002 python nfl_picker_api.py
```

## API Endpoints

### Health Check
- `GET /` - Returns API status and available endpoints

### Team Statistics
- `GET /api/nfl-picker/stats?team1={team1}&team2={team2}`
  - Returns statistics for two teams
  - Example: `GET /api/nfl-picker/stats?team1=Kansas City Chiefs&team2=Buffalo Bills`

### Run Analysis
- `POST /api/nfl-picker/analyze`
  - Runs CrewAI analysis for two teams
  - Request body:
    ```json
    {
      "team1": "Kansas City Chiefs",
      "team2": "Buffalo Bills",
      "homeTeam": "Buffalo Bills",
      "includeInjuries": true,
      "includeCoaching": true,
      "includeSpecialTeams": true
    }
    ```
  - **Note**: This may take several minutes to complete as it runs full AI analysis

### Submit Game Result
- `POST /api/nfl-picker/submit-result`
  - Saves a game result to the database
  - Request body:
    ```json
    {
      "week": 1,
      "team1": "Kansas City Chiefs",
      "team2": "Buffalo Bills",
      "winningTeam": "Kansas City Chiefs",
      "team1Score": 24,
      "team2Score": 17
    }
    ```

### Update Database
- `POST /api/nfl-picker/update-database`
  - Triggers database update (stats collection)
  - Returns status message

## Using with the Web GUI

1. Start the API server:
   ```bash
   python nfl_picker_api.py
   ```

2. Open the web GUI:
   - Navigate to `nfl_picker/gui.html` in your browser
   - Or access it through your portfolio site

3. The GUI will automatically connect to the API on `http://localhost:8001`

## Troubleshooting

### Port Already in Use
If port 8001 is already in use, change it:
```bash
NFL_PICKER_PORT=8002 python nfl_picker_api.py
```
Then update the API URL in `nfl_picker/gui.html` to use port 8002.

### Import Errors
Make sure you're running from the project root directory and that the `nfl_picker/src` directory is in your Python path.

### API Key Errors
Ensure your `.env` file contains:
- `OPENAI_API_KEY=your_key_here`
- `SERPER_API_KEY=your_key_here` (for web search)

### Analysis Takes Too Long
The analysis uses CrewAI with multiple AI agents and can take 5-15 minutes depending on the complexity. This is normal behavior.

## Production Deployment

For production:
1. Change CORS settings to allow only your domain
2. Use environment variables for all configuration
3. Set up proper error logging and monitoring
4. Consider using a production ASGI server like Gunicorn with Uvicorn workers
5. Implement rate limiting for API endpoints

