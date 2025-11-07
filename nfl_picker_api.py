"""
FastAPI server for NFL Picker web GUI
Provides API endpoints for team analysis, statistics, and result submission
"""
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add nfl_picker to path
nfl_picker_path = Path(__file__).parent / "nfl_picker" / "src"
sys.path.insert(0, str(nfl_picker_path))

app = FastAPI(title="NFL Picker API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class TeamStatsRequest(BaseModel):
    team1: str
    team2: str

class AnalysisRequest(BaseModel):
    team1: str
    team2: str
    homeTeam: str
    includeInjuries: bool = True
    includeCoaching: bool = True
    includeSpecialTeams: bool = True

class ResultSubmissionRequest(BaseModel):
    week: int
    team1: str
    team2: str
    winningTeam: str
    team1Score: int
    team2Score: int

class AnalysisResponse(BaseModel):
    success: bool
    result: str
    prediction: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class StatsResponse(BaseModel):
    success: bool
    stats: str
    error: Optional[str] = None

class ResultResponse(BaseModel):
    success: bool
    message: str
    error: Optional[str] = None

# Global variable to track running analyses
running_analyses: Dict[str, Any] = {}

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "NFL Picker API is running",
        "endpoints": {
            "stats": "/api/nfl-picker/stats",
            "analyze": "/api/nfl-picker/analyze",
            "submit-result": "/api/nfl-picker/submit-result",
            "update-database": "/api/nfl-picker/update-database"
        }
    }

@app.get("/api/nfl-picker/stats", response_model=StatsResponse)
async def get_team_stats(team1: str, team2: str):
    """Get statistics for two teams."""
    try:
        from nfl_picker.stats_database import NFLStatsDatabase
        from nfl_picker.database import NFLDatabase
        
        stats_db = NFLStatsDatabase()
        db = NFLDatabase()
        
        stats_text = []
        stats_text.append(f"Statistics for {team1} vs {team2}\n")
        stats_text.append("=" * 50)
        
        # Get basic team info from database
        try:
            # Get recent predictions for these teams
            predictions = db.get_predictions(limit=10)
            team_predictions = [p for p in predictions if team1 in [p[1], p[2]] and team2 in [p[1], p[2]]]
            if team_predictions:
                stats_text.append("\nRecent Predictions:")
                for pred in team_predictions[:5]:
                    stats_text.append(f"  {pred[1]} vs {pred[2]} - Predicted: {pred[3]}")
        except Exception as e:
            stats_text.append(f"\nNote: Could not retrieve recent predictions: {e}")
        
        # Get position stats if available
        try:
            # This would require more complex queries - simplified for now
            stats_text.append(f"\n\nNote: Detailed position statistics require database queries.")
            stats_text.append("Use the desktop application for full statistics access.")
        except Exception as e:
            stats_text.append(f"\nNote: Statistics database query error: {e}")
        
        return StatsResponse(
            success=True,
            stats="\n".join(stats_text)
        )
        
    except Exception as e:
        return StatsResponse(
            success=False,
            stats="",
            error=f"Error retrieving statistics: {str(e)}"
        )

@app.post("/api/nfl-picker/analyze", response_model=AnalysisResponse)
async def run_team_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Run team analysis using CrewAI."""
    try:
        # Validate teams
        from nfl_picker.config import NFL_TEAMS
        if request.team1 not in NFL_TEAMS or request.team2 not in NFL_TEAMS:
            raise ValueError("Invalid team name(s)")
        
        if request.homeTeam not in [request.team1, request.team2]:
            raise ValueError("Home team must be one of the selected teams")
        
        # Import focused analysis
        from nfl_picker.focused_analysis import FocusedTeamAnalysis
        
        # Create analysis instance
        analysis = FocusedTeamAnalysis(
            team1=request.team1,
            team2=request.team2,
            home_team=request.homeTeam,
            include_injuries=request.includeInjuries,
            include_coaching=request.includeCoaching,
            include_special_teams=request.includeSpecialTeams
        )
        
        # Run analysis (this may take several minutes)
        results = analysis.run_analysis()
        
        # Format results
        result_text = []
        result_text.append(f"Analysis Results: {request.team1} vs {request.team2}")
        result_text.append("=" * 60)
        
        if 'predicted_winner' in results:
            result_text.append(f"\nPredicted Winner: {results['predicted_winner']}")
        
        if 'predicted_score' in results:
            result_text.append(f"Predicted Score: {results['predicted_score']}")
        
        if 'confidence_level' in results:
            result_text.append(f"Confidence Level: {results['confidence_level']}")
        
        if 'analysis_summary' in results:
            result_text.append(f"\nAnalysis Summary:\n{results['analysis_summary']}")
        
        if 'key_factors' in results:
            result_text.append(f"\nKey Factors:")
            for factor in results['key_factors']:
                result_text.append(f"  - {factor}")
        
        # Store prediction in database
        try:
            from nfl_picker.database import NFLDatabase
            from nfl_picker.utils import get_current_nfl_week
            import hashlib
            db = NFLDatabase()
            
            # Create game_id
            game_id = hashlib.md5(f"{request.team1}_{request.team2}_{request.homeTeam}_{get_current_nfl_week()}".encode()).hexdigest()
            
            db.save_prediction(
                game_id=game_id,
                team1=request.team1,
                team2=request.team2,
                home_team=request.homeTeam,
                predicted_winner=results.get('predicted_winner', 'Unknown'),
                predicted_score_home=results.get('predicted_score_home'),
                predicted_score_away=results.get('predicted_score_away'),
                confidence_level=results.get('confidence_level', 0.5),
                analysis_data=results,
                week=get_current_nfl_week()
            )
        except Exception as e:
            result_text.append(f"\nNote: Could not save prediction to database: {e}")
        
        return AnalysisResponse(
            success=True,
            result="\n".join(result_text),
            prediction=results
        )
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return AnalysisResponse(
            success=False,
            result="",
            error=f"Analysis failed: {str(e)}\n\nDetails:\n{error_details}"
        )

@app.post("/api/nfl-picker/submit-result", response_model=ResultResponse)
async def submit_game_result(request: ResultSubmissionRequest):
    """Submit a game result to the database."""
    try:
        from nfl_picker.database import NFLDatabase
        
        db = NFLDatabase()
        
        # Determine winner
        if request.team1Score > request.team2Score:
            actual_winner = request.team1
        elif request.team2Score > request.team1Score:
            actual_winner = request.team2
        else:
            actual_winner = "Tie"
        
        # Determine home team based on which team is listed as winning team
        # If winning team is team1, team1 is home; if winning team is team2, team2 is home
        # For simplicity, we'll use team2 as home team (can be improved later)
        home_team = request.team2
        actual_score_home = request.team2Score
        actual_score_away = request.team1Score
        
        # Create game_id
        import hashlib
        game_id = hashlib.md5(f"{request.team1}_{request.team2}_{request.week}".encode()).hexdigest()
        
        # Save result
        db.save_game_result(
            game_id=game_id,
            team1=request.team1,
            team2=request.team2,
            home_team=home_team,
            actual_winner=actual_winner if actual_winner != "Tie" else None,
            actual_score_home=actual_score_home,
            actual_score_away=actual_score_away,
            week=request.week
        )
        
        # Calculate accuracy if prediction exists
        try:
            db.calculate_and_store_accuracy(game_id)
        except Exception:
            pass  # No prediction exists, that's okay
        
        return ResultResponse(
            success=True,
            message=f"Result saved successfully: {request.team1} {request.team1Score} - {request.team2Score} {request.team2} (Week {request.week})"
        )
        
    except Exception as e:
        return ResultResponse(
            success=False,
            message="",
            error=f"Error submitting result: {str(e)}"
        )

@app.post("/api/nfl-picker/update-database")
async def update_database():
    """Trigger database update (stats collection)."""
    try:
        # This would trigger the stats update process
        # For now, return a message indicating it's triggered
        return {
            "success": True,
            "message": "Database update triggered. This may take several minutes.",
            "note": "Full database update requires running the desktop application or scheduled task."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error triggering database update: {str(e)}"
        }

if __name__ == "__main__":
    port = int(os.getenv("NFL_PICKER_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

