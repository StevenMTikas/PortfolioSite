"""
NFL Picker GUI Application
Allows selection of two teams for focused analysis and game prediction
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
from pathlib import Path
from datetime import datetime
import re

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nfl_picker.config import NFL_TEAMS
from nfl_picker.database import NFLDatabase
from nfl_picker.stats_database import NFLStatsDatabase
from nfl_picker.utils import get_current_nfl_week


class NFLEamSelector:
    """GUI for selecting NFL teams and running focused analysis."""

    def __init__(self, root):
        self.root = root
        self.root.title("NFL Picker - Team Analysis & Learning")
        self.root.geometry("1200x900")
        self.root.configure(bg='#1e3a8a')

        # Use centralized NFL teams list
        self.nfl_teams = NFL_TEAMS

        self.current_results = None
        self.db = NFLDatabase()
        self.stats_db = NFLStatsDatabase()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#1e3a8a', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="NFL Picker - Team Analysis & Learning", 
            font=('Arial', 20, 'bold'),
            fg='white',
            bg='#1e3a8a'
        )
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Configure notebook style for bigger tabs
        style = ttk.Style()
        style.configure('TNotebook.Tab', padding=[20, 10], font=('Arial', 12, 'bold'))
        
        # Create tabs
        self.analysis_frame = tk.Frame(self.notebook, bg='#1e3a8a')
        self.results_frame = tk.Frame(self.notebook, bg='#1e3a8a')
        
        self.notebook.add(self.analysis_frame, text="Team Analysis")
        self.notebook.add(self.results_frame, text="Result Entry")
        
        # Setup analysis tab
        self.setup_analysis_tab()
        
        # Setup results entry tab
        self.setup_results_tab()
    
    def setup_analysis_tab(self):
        """Setup the team analysis tab."""
        # Team selection frame
        team_frame = tk.Frame(self.analysis_frame, bg='#1e3a8a')
        team_frame.pack(fill=tk.X, pady=10)
        
        # Team 1 selection
        tk.Label(team_frame, text="Team 1:", font=('Arial', 12, 'bold'), fg='white', bg='#1e3a8a').pack(anchor='w')
        self.team1_var = tk.StringVar()
        self.team1_combo = ttk.Combobox(team_frame, textvariable=self.team1_var, values=self.nfl_teams, width=35, font=('Arial', 11))
        self.team1_combo.pack(fill=tk.X, pady=(5, 15))
        
        # Team 2 selection
        tk.Label(team_frame, text="Team 2:", font=('Arial', 12, 'bold'), fg='white', bg='#1e3a8a').pack(anchor='w')
        self.team2_var = tk.StringVar()
        self.team2_combo = ttk.Combobox(team_frame, textvariable=self.team2_var, values=self.nfl_teams, width=35, font=('Arial', 11))
        self.team2_combo.pack(fill=tk.X, pady=(5, 15))
        
        # Team stats display section
        stats_frame = tk.Frame(self.analysis_frame, bg='#1e3a8a')
        stats_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(stats_frame, text="Team Statistics:", font=('Arial', 12, 'bold'), fg='white', bg='#1e3a8a').pack(anchor='w')
        
        # Stats display area
        self.stats_text = tk.Text(stats_frame, height=8, width=80, font=('Courier', 9), 
                                 bg='#f8f9fa', fg='#212529', wrap=tk.WORD)
        self.stats_text.pack(fill=tk.X, pady=(5, 10))
        
        # Stats refresh button
        stats_button_frame = tk.Frame(stats_frame, bg='#1e3a8a')
        stats_button_frame.pack(fill=tk.X)
        
        self.refresh_stats_button = tk.Button(stats_button_frame, text="Refresh Team Stats", 
                                            command=self.refresh_team_stats, 
                                            bg='#10b981', fg='white', font=('Arial', 10, 'bold'),
                                            padx=20, pady=5)
        self.refresh_stats_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.update_stats_button = tk.Button(stats_button_frame, text="Update Database", 
                                           command=self.update_stats_database, 
                                           bg='#f59e0b', fg='white', font=('Arial', 10, 'bold'),
                                           padx=20, pady=5)
        self.update_stats_button.pack(side=tk.LEFT)
        
        # Bind team selection changes to refresh stats
        self.team1_combo.bind('<<ComboboxSelected>>', lambda e: self.refresh_team_stats())
        self.team2_combo.bind('<<ComboboxSelected>>', lambda e: self.refresh_team_stats())
        
        # Analysis options
        options_frame = tk.Frame(self.analysis_frame, bg='#1e3a8a')
        options_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(options_frame, text="Analysis Options:", font=('Arial', 12, 'bold'), fg='white', bg='#1e3a8a').pack(anchor='w')
        
        # Checkboxes for analysis types
        self.include_injuries = tk.BooleanVar(value=True)
        self.include_coaching = tk.BooleanVar(value=True)
        self.include_special_teams = tk.BooleanVar(value=True)
        
        tk.Checkbutton(options_frame, text="Include Injury Analysis", variable=self.include_injuries, 
                      fg='white', bg='#1e3a8a', selectcolor='#3b82f6').pack(anchor='w')
        tk.Checkbutton(options_frame, text="Include Coaching Analysis", variable=self.include_coaching, 
                      fg='white', bg='#1e3a8a', selectcolor='#3b82f6').pack(anchor='w')
        tk.Checkbutton(options_frame, text="Include Special Teams Analysis", variable=self.include_special_teams, 
                      fg='white', bg='#1e3a8a', selectcolor='#3b82f6').pack(anchor='w')
        
        # Buttons frame
        button_frame = tk.Frame(self.analysis_frame, bg='#1e3a8a')
        button_frame.pack(fill=tk.X, pady=20)
        
        # Run analysis button
        self.run_button = tk.Button(
            button_frame,
            text="Run Team Analysis",
            command=self.run_analysis,
            bg='#3b82f6',
            fg='white',
            font=('Arial', 14, 'bold'),
            padx=30,
            pady=15,
            relief=tk.RAISED,
            bd=2,
            width=18,
            height=2
        )
        self.run_button.pack(side=tk.LEFT, padx=(0, 15))
        
        # Clear button
        clear_button = tk.Button(
            button_frame,
            text="Clear All",
            command=self.clear_selections,
            bg='#6b7280',
            fg='white',
            font=('Arial', 14, 'bold'),
            padx=30,
            pady=15,
            relief=tk.RAISED,
            bd=2,
            width=18,
            height=2
        )
        clear_button.pack(side=tk.LEFT, padx=(0, 15))
        
        
        # Progress frame
        progress_frame = tk.Frame(self.analysis_frame, bg='#1e3a8a')
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_var = tk.StringVar(value="Ready to analyze teams")
        self.progress_label = tk.Label(progress_frame, textvariable=self.progress_var, 
                                     fg='white', bg='#1e3a8a', font=('Arial', 10))
        self.progress_label.pack(anchor='w')
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Results frame
        results_frame = tk.Frame(self.analysis_frame, bg='#1e3a8a')
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tk.Label(results_frame, text="Analysis Results:", font=('Arial', 12, 'bold'), 
                fg='white', bg='#1e3a8a').pack(anchor='w')
        
        # Results text area
        self.results_text = tk.Text(results_frame, height=12, width=100, wrap=tk.WORD, 
                                   bg='#1f2937', fg='white', font=('Consolas', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Scrollbar for results
        scrollbar = tk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
    
    def setup_results_tab(self):
        """Setup the result entry and learning tab."""
        # Title for results tab
        title_label = tk.Label(
            self.results_frame, 
            text="Enter Actual Game Results for Learning", 
            font=('Arial', 16, 'bold'),
            fg='white',
            bg='#1e3a8a'
        )
        title_label.pack(pady=(0, 20))
        
        # Game selection frame
        game_frame = tk.Frame(self.results_frame, bg='#1e3a8a')
        game_frame.pack(fill=tk.X, pady=10)
        
        # Game selection
        tk.Label(game_frame, text="Select Game to Enter Results:", font=('Arial', 12, 'bold'), fg='white', bg='#1e3a8a').pack(anchor='w')
        
        # Get predictions from database
        self.prediction_var = tk.StringVar()
        self.prediction_combo = ttk.Combobox(game_frame, textvariable=self.prediction_var, width=50, font=('Arial', 11))
        self.prediction_combo.pack(fill=tk.X, pady=(5, 15))
        self.update_prediction_list()
        
        # Refresh button
        refresh_button = tk.Button(
            game_frame,
            text="Refresh Game List",
            command=self.update_prediction_list,
            bg='#6b7280',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=5
        )
        refresh_button.pack(anchor='w')
        
        # Result entry frame
        result_frame = tk.Frame(self.results_frame, bg='#1e3a8a')
        result_frame.pack(fill=tk.X, pady=20)
        
        tk.Label(result_frame, text="Enter Actual Game Results:", font=('Arial', 12, 'bold'), fg='white', bg='#1e3a8a').pack(anchor='w')
        
        # Score entry frame
        score_frame = tk.Frame(result_frame, bg='#1e3a8a')
        score_frame.pack(fill=tk.X, pady=10)
        
        # Home team score
        home_score_frame = tk.Frame(score_frame, bg='#1e3a8a')
        home_score_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(home_score_frame, text="Home Team Score:", font=('Arial', 10, 'bold'), fg='white', bg='#1e3a8a').pack(anchor='w')
        self.home_score_var = tk.StringVar()
        self.home_score_entry = tk.Entry(home_score_frame, textvariable=self.home_score_var, width=10, font=('Arial', 12))
        self.home_score_entry.pack(pady=(5, 0))
        
        # Away team score
        away_score_frame = tk.Frame(score_frame, bg='#1e3a8a')
        away_score_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(away_score_frame, text="Away Team Score:", font=('Arial', 10, 'bold'), fg='white', bg='#1e3a8a').pack(anchor='w')
        self.away_score_var = tk.StringVar()
        self.away_score_entry = tk.Entry(away_score_frame, textvariable=self.away_score_var, width=10, font=('Arial', 12))
        self.away_score_entry.pack(pady=(5, 0))
        
        # Additional game info frame
        info_frame = tk.Frame(result_frame, bg='#1e3a8a')
        info_frame.pack(fill=tk.X, pady=10)
        
        # Game date
        date_frame = tk.Frame(info_frame, bg='#1e3a8a')
        date_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(date_frame, text="Game Date (YYYY-MM-DD):", font=('Arial', 10, 'bold'), fg='white', bg='#1e3a8a').pack(anchor='w')
        self.game_date_var = tk.StringVar()
        self.game_date_entry = tk.Entry(date_frame, textvariable=self.game_date_var, width=15, font=('Arial', 12))
        self.game_date_entry.pack(pady=(5, 0))
        
        # Weather conditions
        weather_frame = tk.Frame(info_frame, bg='#1e3a8a')
        weather_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(weather_frame, text="Weather Conditions:", font=('Arial', 10, 'bold'), fg='white', bg='#1e3a8a').pack(anchor='w')
        self.weather_var = tk.StringVar()
        self.weather_entry = tk.Entry(weather_frame, textvariable=self.weather_var, width=20, font=('Arial', 12))
        self.weather_entry.pack(pady=(5, 0))
        
        
        # Buttons frame
        button_frame = tk.Frame(self.results_frame, bg='#1e3a8a')
        button_frame.pack(fill=tk.X, pady=20)
        
        # Save result button
        save_result_button = tk.Button(
            button_frame,
            text="Save Game Result",
            command=self.save_game_result,
            bg='#10b981',
            fg='white',
            font=('Arial', 14, 'bold'),
            padx=30,
            pady=15,
            relief=tk.RAISED,
            bd=2,
            width=20,
            height=2
        )
        save_result_button.pack(side=tk.LEFT, padx=(0, 15))
        
        # View accuracy button
        accuracy_button = tk.Button(
            button_frame,
            text="View Prediction Accuracy",
            command=self.view_accuracy,
            bg='#3b82f6',
            fg='white',
            font=('Arial', 14, 'bold'),
            padx=30,
            pady=15,
            relief=tk.RAISED,
            bd=2,
            width=20,
            height=2
        )
        accuracy_button.pack(side=tk.LEFT, padx=(0, 15))
        
        # Clear result form button
        clear_result_button = tk.Button(
            button_frame,
            text="Clear Form",
            command=self.clear_result_form,
            bg='#6b7280',
            fg='white',
            font=('Arial', 14, 'bold'),
            padx=30,
            pady=15,
            relief=tk.RAISED,
            bd=2,
            width=20,
            height=2
        )
        clear_result_button.pack(side=tk.LEFT)
        
        # Status frame
        status_frame = tk.Frame(self.results_frame, bg='#1e3a8a')
        status_frame.pack(fill=tk.X, pady=10)
        
        self.result_status_var = tk.StringVar(value="Ready to enter game results")
        self.result_status_label = tk.Label(status_frame, textvariable=self.result_status_var, 
                                          fg='white', bg='#1e3a8a', font=('Arial', 10))
        self.result_status_label.pack(anchor='w')
        
        # Learning data display frame
        learning_frame = tk.Frame(self.results_frame, bg='#1e3a8a')
        learning_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tk.Label(learning_frame, text="Learning Data & Accuracy:", font=('Arial', 12, 'bold'), 
                fg='white', bg='#1e3a8a').pack(anchor='w')
        
        # Learning data text area
        self.learning_text = tk.Text(learning_frame, height=12, width=100, wrap=tk.WORD, 
                                    bg='#1f2937', fg='white', font=('Consolas', 10))
        self.learning_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Scrollbar for learning data
        learning_scrollbar = tk.Scrollbar(learning_frame, orient=tk.VERTICAL, command=self.learning_text.yview)
        learning_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.learning_text.config(yscrollcommand=learning_scrollbar.set)
        
    def clear_selections(self):
        """Clear all selections."""
        self.team1_var.set("")
        self.team2_var.set("")
        self.results_text.delete(1.0, tk.END)
        self.progress_var.set("Ready to analyze teams")
        self.current_results = None
        
    def validate_selections(self):
        """Validate that all required fields are filled."""
        if not self.team1_var.get():
            messagebox.showerror("Error", "Please select Team 1")
            return False
        if not self.team2_var.get():
            messagebox.showerror("Error", "Please select Team 2")
            return False
        if self.team1_var.get() == self.team2_var.get():
            messagebox.showerror("Error", "Team 1 and Team 2 must be different")
            return False
        return True
        
    def run_analysis(self):
        """Run the focused team analysis."""
        if not self.validate_selections():
            return
            
        # Disable button during analysis
        self.run_button.config(state='disabled')
        self.progress_bar.start()
        
        try:
            # Import the focused analysis module
            from nfl_picker.focused_analysis import FocusedTeamAnalysis
            
            # Get selections
            team1 = self.team1_var.get()
            team2 = self.team2_var.get()
            home_team = team2  # Team 2 is always the home team
            
            # Update progress
            self.progress_var.set(f"Analyzing {team1} vs {team2}...")
            self.root.update()
            
            # Create focused analysis
            analysis = FocusedTeamAnalysis(
                team1=team1,
                team2=team2,
                home_team=home_team,
                include_injuries=self.include_injuries.get(),
                include_coaching=self.include_coaching.get(),
                include_special_teams=self.include_special_teams.get()
            )
            
            # Run analysis
            results = analysis.run_analysis()
            
            # Display results
            self.display_results(results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.progress_var.set("Analysis failed")
        finally:
            # Re-enable button
            self.run_button.config(state='normal')
            self.progress_bar.stop()
            
    def display_results(self, results):
        """Display analysis results and automatically save to output folder."""
        self.current_results = results
        self.results_text.delete(1.0, tk.END)
        
        # Format results
        output = f"=== NFL TEAM ANALYSIS RESULTS ===\n\n"
        output += f"Teams: {results['team1']} vs {results['team2']}\n"
        output += f"Home Team: {results['home_team']}\n"
        output += f"Analysis Date: {results['analysis_date']}\n\n"
        
        output += f"=== PREDICTION ===\n"
        output += f"Predicted Winner: {results['predicted_winner']}\n"
        output += f"Predicted Score: {results['predicted_score']}\n"
        output += f"Confidence: {results['confidence']}\n\n"
        
        output += f"=== KEY FACTORS ===\n"
        for factor in results['key_factors']:
            output += f"• {factor}\n"
        output += "\n"
        
        # Add team statistics section
        output += f"=== TEAM STATISTICS ===\n"
        try:
            week = get_current_nfl_week()
            season = 2025
            
            # Get team statistics
            team1_stats = self.get_team_stats_summary(results['team1'], week, season)
            team2_stats = self.get_team_stats_summary(results['team2'], week, season)
            
            output += f"\n--- {results['team1']} ---\n"
            output += team1_stats + "\n"
            
            output += f"--- {results['team2']} ---\n"
            output += team2_stats + "\n"
            
        except Exception as e:
            output += f"Statistics not available: {str(e)}\n\n"
        
        output += f"=== DETAILED ANALYSIS ===\n"
        output += results['detailed_analysis']
        
        self.results_text.insert(1.0, output)
        self.progress_var.set("Analysis completed successfully")
        
        # Automatically save the results to output folder
        self.auto_save_results()
    

    
    def save_prediction_to_database(self, week):
        """Save prediction to database for learning purposes."""
        try:
            # Create unique game ID
            team1_clean = self.current_results['team1'].replace(" ", "_").replace(".", "")
            team2_clean = self.current_results['team2'].replace(" ", "_").replace(".", "")
            game_id = f"{team1_clean}_vs_{team2_clean}_Week_{week:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Extract prediction data
            predicted_winner = self.current_results['predicted_winner']
            predicted_score = self.current_results['predicted_score']

            # Parse predicted score (format: "Eagles 30, Giants 17")
            try:
                if " vs " in predicted_score or " - " in predicted_score:
                    # Handle different score formats
                    score_parts = predicted_score.replace(" vs ", " ").replace(" - ", " ").split()
                    if len(score_parts) >= 4:
                        home_score = int(score_parts[1].rstrip(','))
                        away_score = int(score_parts[3])
                    else:
                        home_score = away_score = 0
                else:
                    # Try to extract numbers from score string
                    numbers = re.findall(r'\d+', predicted_score)
                    if len(numbers) >= 2:
                        home_score = int(numbers[0])
                        away_score = int(numbers[1])
                    else:
                        home_score = away_score = 0
            except:
                home_score = away_score = 0

            # Extract confidence level
            confidence = 0.75  # Default confidence (changed from 0.85 to match actual analysis)
            if 'confidence' in self.current_results:
                try:
                    conf_text = str(self.current_results['confidence'])
                    # Try to extract percentage
                    conf_match = re.search(r'(\d+)%', conf_text)
                    if conf_match:
                        confidence = float(conf_match.group(1)) / 100
                    else:
                        # Try to extract decimal format (e.g., 0.75)
                        decimal_match = re.search(r'0\.\d+', conf_text)
                        if decimal_match:
                            confidence = float(decimal_match.group(0))
                except Exception as e:
                    print(f"Error parsing confidence: {e}, using default")

            # Store prediction using database manager
            self.db.save_prediction(
                game_id=game_id,
                team1=self.current_results['team1'],
                team2=self.current_results['team2'],
                home_team=self.current_results['team2'],  # Team 2 is always the home team
                predicted_winner=predicted_winner,
                predicted_score_home=home_score,
                predicted_score_away=away_score,
                confidence_level=confidence,
                analysis_data=self.current_results,
                week=week,
                season=2025
            )

        except Exception as e:
            print(f"Error saving prediction to database: {e}")
    
    def auto_save_results(self):
        """Automatically save analysis results to output folder when analysis completes."""
        if not self.current_results:
            return

        try:
            # Get current week using utility function
            week = get_current_nfl_week()
            
            # Create filename with both teams and week
            team1_clean = self.current_results['team1'].replace(" ", "_").replace(".", "")
            team2_clean = self.current_results['team2'].replace(" ", "_").replace(".", "")
            filename = f"NFL_Analysis_{team1_clean}_vs_{team2_clean}_Week_{week:02d}.txt"
            
            # Get the content from the results text area
            content = self.results_text.get(1.0, tk.END)
            
            # Add header with file info
            file_content = f"NFL Picker Analysis Results\n"
            file_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            file_content += f"Week: {week}\n"
            file_content += f"Season: 2025\n"
            file_content += "=" * 50 + "\n\n"
            file_content += content
            
            # Create output directory if it doesn't exist
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            # Save file
            file_path = output_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            # Save to database for learning
            self.save_prediction_to_database(week)
            
            # Update progress to show auto-save
            self.progress_var.set(f"Analysis completed and auto-saved to {filename}")
            
        except Exception as e:
            print(f"Error auto-saving results: {e}")
            # Don't show error to user since this is automatic
    
    def update_prediction_list(self):
        """Update the list of predictions in the combo box."""
        try:
            predictions = self.db.get_predictions()

            prediction_list = []
            for pred in predictions:
                game_id, team1, team2, winner, date = pred
                display_text = f"{team1} vs {team2} (Predicted: {winner}) - {date}"
                prediction_list.append(display_text)

            self.prediction_combo['values'] = prediction_list
            if hasattr(self, 'result_status_var') and self.result_status_var:
                self.result_status_var.set(f"Found {len(prediction_list)} predictions")

        except Exception as e:
            if hasattr(self, 'result_status_var') and self.result_status_var:
                self.result_status_var.set(f"Error loading predictions: {str(e)}")
    
    def save_game_result(self):
        """Save the actual game result to the database."""
        if not self.prediction_var.get():
            messagebox.showerror("Error", "Please select a game first")
            return
        
        try:
            # Get the selected prediction
            selected_text = self.prediction_var.get()
            game_id = selected_text.split(' - ')[0].split(' (Predicted: ')[0]
            
            # Validate scores
            home_score = self.home_score_var.get().strip()
            away_score = self.away_score_var.get().strip()
            
            if not home_score or not away_score:
                messagebox.showerror("Error", "Please enter both home and away scores")
                return
            
            try:
                home_score = int(home_score)
                away_score = int(away_score)
            except ValueError:
                messagebox.showerror("Error", "Scores must be numbers")
                return
            
            # Determine actual winner
            if home_score > away_score:
                actual_winner = self.db.get_team_from_prediction(game_id, 'home_team')
            elif away_score > home_score:
                team1 = self.db.get_team_from_prediction(game_id, 'team1')
                team2 = self.db.get_team_from_prediction(game_id, 'team2')
                home_team = self.db.get_team_from_prediction(game_id, 'home_team')
                actual_winner = team2 if team1 == home_team else team1
            else:
                actual_winner = "Tie"

            # Get additional info
            game_date = self.game_date_var.get().strip() or None
            weather = self.weather_var.get().strip() or None

            # Get current week and season
            week = get_current_nfl_week()
            season = 2025

            # Save game result using database manager
            self.db.save_game_result(
                game_id=game_id,
                team1=self.db.get_team_from_prediction(game_id, 'team1'),
                team2=self.db.get_team_from_prediction(game_id, 'team2'),
                home_team=self.db.get_team_from_prediction(game_id, 'home_team'),
                actual_winner=actual_winner,
                actual_score_home=home_score,
                actual_score_away=away_score,
                game_date=game_date,
                weather_conditions=weather,
                week=week,
                season=season
            )

            # Calculate and store accuracy
            self.db.calculate_and_store_accuracy(game_id)
            messagebox.showinfo("Success", "Game result saved successfully!")
            self.clear_result_form()
            self.update_prediction_list()
            self.view_accuracy()  # Refresh accuracy display
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save game result: {str(e)}")
    

    
    def view_accuracy(self):
        """Display prediction accuracy statistics."""
        try:
            stats = self.db.get_accuracy_stats()
            if not stats or stats[0] == 0:
                self.learning_text.delete(1.0, tk.END)
                self.learning_text.insert(1.0, "No accuracy data available yet. Enter some game results to see learning progress.")
                return

            total, correct, avg_conf, avg_diff = stats
            accuracy_rate = (correct / total) * 100 if total > 0 else 0

            # Get recent predictions with results
            recent_games = self.db.get_recent_games_with_results(limit=10)
            
            # Format output
            output = "=== PREDICTION ACCURACY STATISTICS ===\n\n"
            output += f"Total Predictions: {total}\n"
            output += f"Correct Predictions: {correct}\n"
            output += f"Accuracy Rate: {accuracy_rate:.1f}%\n"
            output += f"Average Confidence vs Accuracy: {avg_conf:.2f}\n"
            output += f"Average Score Difference: {avg_diff:.1f} points\n\n"
            
            output += "=== RECENT GAME RESULTS ===\n\n"
            for game in recent_games:
                team1, team2, pred_winner, conf, actual_winner, home_score, away_score, correct, diff = game
                result_icon = "✓" if correct else "✗"
                output += f"{result_icon} {team1} vs {team2}\n"
                output += f"   Predicted: {pred_winner} (Confidence: {conf:.1f})\n"
                output += f"   Actual: {actual_winner} ({home_score}-{away_score})\n"
                output += f"   Score Difference: {diff} points\n\n"
            
            self.learning_text.delete(1.0, tk.END)
            self.learning_text.insert(1.0, output)
            self.result_status_var.set(f"Accuracy: {accuracy_rate:.1f}% ({correct}/{total})")
            
        except Exception as e:
            self.learning_text.delete(1.0, tk.END)
            self.learning_text.insert(1.0, f"Error loading accuracy data: {str(e)}")
    
    def clear_result_form(self):
        """Clear the result entry form."""
        self.home_score_var.set("")
        self.away_score_var.set("")
        self.game_date_var.set("")
        self.weather_var.set("")
        self.result_status_var.set("Form cleared")
    
    def refresh_team_stats(self):
        """Refresh team statistics display based on selected teams."""
        team1 = self.team1_var.get()
        team2 = self.team2_var.get()
        
        if not team1 and not team2:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, "Select teams to view statistics\n\nNote: Free API tier provides player rosters only.\nDetailed statistics require API upgrade.")
            return
        
        try:
            week = get_current_nfl_week()
            season = 2025
            
            stats_output = f"=== TEAM STATISTICS - WEEK {week} ===\n\n"
            
            if team1:
                stats_output += f"--- {team1} ---\n"
                team1_stats = self.get_team_stats_summary(team1, week, season)
                stats_output += team1_stats + "\n"
            
            if team2:
                stats_output += f"--- {team2} ---\n"
                team2_stats = self.get_team_stats_summary(team2, week, season)
                stats_output += team2_stats + "\n"
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats_output)
            
        except Exception as e:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, f"Error loading team statistics: {str(e)}")
    
    def get_team_stats_summary(self, team_name, week, season):
        """Get a summary of team statistics for display."""
        try:
            # Use full team name directly (database stores full names)
            # Get all position group stats
            team_stats = self.stats_db.get_team_stats(team_name, week, season)
            
            summary = ""
            
            # Quarterback stats
            qb_stats = team_stats.get('QB', [])
            if qb_stats:
                qb = qb_stats[0]  # Get top QB
                summary += f"QB: {qb['player_name']} - "
                summary += f"Rating: {qb.get('passer_rating', 'N/A')}, "
                summary += f"Yards: {qb.get('passing_yards', 0)}, "
                summary += f"TDs: {qb.get('passing_touchdowns', 0)}\n"
            
            # Running back stats
            rb_stats = team_stats.get('RB', [])
            if rb_stats:
                top_rb = rb_stats[0]  # Get top RB
                summary += f"RB: {top_rb['player_name']} - "
                summary += f"Yards: {top_rb.get('rushing_yards', 0)}, "
                summary += f"Attempts: {top_rb.get('rushing_attempts', 0)}, "
                summary += f"TDs: {top_rb.get('rushing_touchdowns', 0)}\n"
            
            # Wide receiver stats
            wr_stats = team_stats.get('WR', [])
            if wr_stats:
                top_wr = wr_stats[0]  # Get top WR
                summary += f"WR: {top_wr['player_name']} - "
                summary += f"Receptions: {top_wr.get('receptions', 0)}, "
                summary += f"Yards: {top_wr.get('receiving_yards', 0)}, "
                summary += f"TDs: {top_wr.get('touchdowns', 0)}\n"
            
            # Defensive stats
            dl_stats = team_stats.get('DL', [])
            if dl_stats:
                top_dl = dl_stats[0]  # Get top DL
                summary += f"DL: {top_dl['player_name']} - "
                summary += f"Tackles: {top_dl.get('tackles', 0)}, "
                summary += f"Sacks: {top_dl.get('sacks', 0)}\n"
            
            lb_stats = team_stats.get('LB', [])
            if lb_stats:
                top_lb = lb_stats[0]  # Get top LB
                summary += f"LB: {top_lb['player_name']} - "
                summary += f"Tackles: {top_lb.get('tackles', 0)}, "
                summary += f"Sacks: {top_lb.get('sacks', 0)}\n"
            
            if not any([qb_stats, rb_stats, wr_stats, dl_stats, lb_stats]):
                summary = "No statistics available for this week"
            else:
                # Check if we have actual statistical data (not just player info)
                has_real_stats = False
                for position_group in [qb_stats, rb_stats, wr_stats, dl_stats, lb_stats]:
                    if position_group:
                        player = position_group[0]
                        # Check if any meaningful stats are > 0
                        stat_fields = ['passing_yards', 'rushing_yards', 'receiving_yards', 'tackles', 'sacks']
                        if any(player.get(field, 0) > 0 for field in stat_fields):
                            has_real_stats = True
                            break
                
                if not has_real_stats:
                    summary += "\nNote: Detailed statistics require API upgrade (free tier provides player rosters only)"
            
            return summary
            
        except Exception as e:
            return f"Error retrieving stats: {str(e)}"
    
    def get_team_abbreviation(self, team_name):
        """Convert full team name to abbreviation."""
        team_mapping = {
            'Arizona Cardinals': 'ARI',
            'Atlanta Falcons': 'ATL',
            'Baltimore Ravens': 'BAL',
            'Buffalo Bills': 'BUF',
            'Carolina Panthers': 'CAR',
            'Chicago Bears': 'CHI',
            'Cincinnati Bengals': 'CIN',
            'Cleveland Browns': 'CLE',
            'Dallas Cowboys': 'DAL',
            'Denver Broncos': 'DEN',
            'Detroit Lions': 'DET',
            'Green Bay Packers': 'GB',
            'Houston Texans': 'HOU',
            'Indianapolis Colts': 'IND',
            'Jacksonville Jaguars': 'JAX',
            'Kansas City Chiefs': 'KC',
            'Las Vegas Raiders': 'LV',
            'Los Angeles Chargers': 'LAC',
            'Los Angeles Rams': 'LAR',
            'Miami Dolphins': 'MIA',
            'Minnesota Vikings': 'MIN',
            'New England Patriots': 'NE',
            'New Orleans Saints': 'NO',
            'New York Giants': 'NYG',
            'New York Jets': 'NYJ',
            'Philadelphia Eagles': 'PHI',
            'Pittsburgh Steelers': 'PIT',
            'San Francisco 49ers': 'SF',
            'Seattle Seahawks': 'SEA',
            'Tampa Bay Buccaneers': 'TB',
            'Tennessee Titans': 'TEN',
            'Washington Commanders': 'WAS'
        }
        return team_mapping.get(team_name)
    
    def update_stats_database(self):
        """Update the statistics database with latest data."""
        try:
            import subprocess
            import sys
            
            # Show progress
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, "Updating statistics database...\nThis may take a few minutes.")
            self.root.update()
            
            # Run the update script
            week = get_current_nfl_week()
            result = subprocess.run([
                sys.executable, 'src/nfl_picker/update_stats.py', 
                '--week', str(week), '--season', '2025', '--api-only'
            ], capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(1.0, f"Database updated successfully for week {week}!\n\nNote: Free API tier provides player rosters only.\nDetailed statistics (yards, TDs, etc.) require API upgrade.\n\nOutput:\n{result.stdout}")
            else:
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(1.0, f"Update failed:\n{result.stderr}")
            
            # Refresh stats display
            self.refresh_team_stats()
            
        except Exception as e:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, f"Error updating database: {str(e)}")

def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = NFLEamSelector(root)
    root.mainloop()

if __name__ == "__main__":
    main()
