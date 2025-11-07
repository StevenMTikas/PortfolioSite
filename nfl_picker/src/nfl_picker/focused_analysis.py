"""
Focused Team Analysis Module
Runs analysis for only two selected teams and provides game prediction
"""

from datetime import datetime
from typing import Dict, List, Any, Tuple

# Apply SSL fix
from nfl_picker.ssl_fix import apply_ssl_fix
apply_ssl_fix()

from nfl_picker.config import CONFIG
from nfl_picker.tools.serper_tool import SerperTool
from nfl_picker.utils import create_analysis_inputs
from crewai import Agent, Crew, Process, Task

class FocusedTeamAnalysis:
    """Focused analysis for two specific NFL teams."""

    # Position groups for analysis
    POSITION_GROUPS = [
        ('defensive_line', 'Defensive Line'),
        ('linebacker', 'Linebacker'),
        ('cornerback', 'Cornerback'),
        ('safety', 'Safety'),
        ('offensive_line', 'Offensive Line'),
        ('tight_end', 'Tight End'),
        ('wide_receiver', 'Wide Receiver'),
        ('running_back', 'Running Back'),
        ('quarterback', 'Quarterback')
    ]

    def __init__(self, team1: str, team2: str, home_team: str,
                 include_injuries: bool = True, include_coaching: bool = True,
                 include_special_teams: bool = True):
        self.team1 = team1
        self.team2 = team2
        self.home_team = home_team
        self.away_team = team1 if team2 == home_team else team2
        self.include_injuries = include_injuries
        self.include_coaching = include_coaching
        self.include_special_teams = include_special_teams

        # Create focused inputs using utility function
        self.inputs = create_analysis_inputs(
            team1=self.team1,
            team2=self.team2,
            home_team=self.home_team,
            include_injuries=str(self.include_injuries),
            include_coaching=str(self.include_coaching),
            include_special_teams=str(self.include_special_teams),
            analysis_type='focused_team_analysis'
        )
        
    def run_analysis(self) -> Dict[str, Any]:
        """Run focused analysis for the two teams."""
        try:
            # Create focused crew
            crew = self.create_focused_crew()
            
            # Run analysis
            results = crew.kickoff(inputs=self.inputs)
            
            # Process results and create prediction
            prediction_results = self.process_results(results)
            
            return prediction_results
            
        except Exception as e:
            raise Exception(f"Focused analysis failed: {str(e)}")
    
    def create_focused_crew(self):
        """Create a focused crew for the two teams."""
        # Create focused agents
        agents = self.create_focused_agents()
        tasks = self.create_focused_tasks(agents)
        
        # Create crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        return crew
    
    def _create_agent(self, role: str, goal: str, backstory: str) -> Agent:
        """
        Factory method to create an agent with standard configuration.

        Args:
            role: Agent role description
            goal: Agent goal description
            backstory: Agent backstory

        Returns:
            Configured Agent instance
        """
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=[SerperTool()],
            verbose=True,
            llm=CONFIG['openai_model']
        )

    def create_focused_agents(self) -> List[Agent]:
        """Create focused agents for the two teams."""
        agents = []

        # Position group agents (focused on the two teams)
        for _, position_name in self.POSITION_GROUPS:
            agent = self._create_agent(
                role=f"Expert {position_name} Analysis for {self.team1} vs {self.team2}",
                goal=f"Analyze {position_name.lower()} for {self.team1} and {self.team2} using 2025 statistics only",
                backstory=f"You are a specialized analyst focusing on {position_name.lower()} for {self.team1} and {self.team2}. "
                         f"Compare their performance using 2025 statistics and identify key differences."
            )
            agents.append(agent)

        # Optional agents based on selections
        if self.include_coaching:
            agents.append(self._create_agent(
                role=f"Expert Coaching Analysis for {self.team1} vs {self.team2}",
                goal=f"Analyze coaching staff effectiveness for {self.team1} and {self.team2} using 2025 data",
                backstory=f"You are a coaching analyst comparing {self.team1} and {self.team2} coaching staffs. "
                         f"Focus on recent performance, game planning, and in-game adjustments."
            ))

        if self.include_special_teams:
            agents.append(self._create_agent(
                role=f"Expert Special Teams Analysis for {self.team1} vs {self.team2}",
                goal=f"Analyze special teams performance for {self.team1} and {self.team2} using 2025 data",
                backstory=f"You are a special teams analyst comparing {self.team1} and {self.team2} special teams units. "
                         f"Focus on kicking, punting, returns, and coverage units."
            ))

        if self.include_injuries:
            agents.append(self._create_agent(
                role=f"Expert Injury Analysis for {self.team1} vs {self.team2}",
                goal=f"Analyze current injuries and their impact on {self.team1} and {self.team2}",
                backstory=f"You are an injury analyst focusing on {self.team1} and {self.team2} current injury reports. "
                         f"Assess how injuries will affect team performance and game outcome."
            ))

        # Home/Away performance agent
        agents.append(self._create_agent(
            role=f"Expert Home/Away Performance Analysis for {self.team1} vs {self.team2}",
            goal=f"Analyze home and away performance for {self.team1} and {self.team2} using 2025 data",
            backstory=f"You are a performance analyst comparing {self.team1} and {self.team2} home and away records. "
                     f"Focus on how location affects their performance."
        ))

        # Team grit and heart agent
        agents.append(self._create_agent(
            role=f"Expert Team Grit Analysis for {self.team1} vs {self.team2}",
            goal=f"Analyze team grit, heart, and intangibles for {self.team1} and {self.team2}",
            backstory=f"You are a team culture analyst comparing {self.team1} and {self.team2} intangibles. "
                     f"Focus on team chemistry, leadership, and mental toughness."
        ))

        return agents
    
    def _create_task(self, description: str, expected_output: str, agent: Agent) -> Task:
        """
        Factory method to create a task with standard configuration.

        Args:
            description: Task description
            expected_output: Expected output description
            agent: Agent to assign the task to

        Returns:
            Configured Task instance
        """
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            llm=CONFIG['openai_model']
        )

    def create_focused_tasks(self, agents: List[Agent]) -> List[Task]:
        """Create focused tasks for the two teams."""
        tasks = []
        task_index = 0

        # Position group tasks
        for _, position_name in self.POSITION_GROUPS:
            if task_index < len(agents):
                task = self._create_task(
                    description=f"Conduct detailed {position_name.lower()} analysis for {self.team1} and {self.team2} using 2025 statistics. "
                               f"Compare their performance, identify strengths and weaknesses, and provide rankings. "
                               f"Focus specifically on these two teams and their head-to-head matchup potential.",
                    expected_output=f"Comprehensive {position_name.lower()} comparison between {self.team1} and {self.team2} "
                                  f"including individual player analysis, unit rankings, and matchup advantages.",
                    agent=agents[task_index]
                )
                tasks.append(task)
                task_index += 1

        # Optional tasks based on selections
        if self.include_coaching and task_index < len(agents):
            tasks.append(self._create_task(
                description=f"Analyze coaching staff effectiveness for {self.team1} and {self.team2} using 2025 data. "
                           f"Compare head coaches, coordinators, and coaching philosophies. "
                           f"Assess game planning, in-game adjustments, and coaching advantages.",
                expected_output=f"Detailed coaching analysis comparing {self.team1} and {self.team2} coaching staffs "
                               f"including strengths, weaknesses, and coaching matchup advantages.",
                agent=agents[task_index]
            ))
            task_index += 1

        if self.include_special_teams and task_index < len(agents):
            tasks.append(self._create_task(
                description=f"Analyze special teams performance for {self.team1} and {self.team2} using 2025 data. "
                           f"Compare kicking, punting, return, and coverage units. "
                           f"Assess special teams impact on field position and game outcomes.",
                expected_output=f"Comprehensive special teams analysis comparing {self.team1} and {self.team2} "
                               f"including unit rankings, key players, and special teams advantages.",
                agent=agents[task_index]
            ))
            task_index += 1

        if self.include_injuries and task_index < len(agents):
            tasks.append(self._create_task(
                description=f"Analyze current injury reports for {self.team1} and {self.team2}. "
                           f"Assess impact of key injuries on team performance. "
                           f"Evaluate depth chart adjustments and injury-related advantages.",
                expected_output=f"Detailed injury analysis for {self.team1} and {self.team2} "
                               f"including current injury status, impact assessment, and injury-related advantages.",
                agent=agents[task_index]
            ))
            task_index += 1

        # Home/Away performance task
        if task_index < len(agents):
            tasks.append(self._create_task(
                description=f"Analyze home and away performance for {self.team1} and {self.team2} using 2025 data. "
                           f"Compare their records at home vs away, identify home field advantages, "
                           f"and assess how location affects their performance.",
                expected_output=f"Comprehensive home/away analysis for {self.team1} and {self.team2} "
                               f"including home field advantages, away game performance, and location impact.",
                agent=agents[task_index]
            ))
            task_index += 1

        # Team grit and heart task
        if task_index < len(agents):
            tasks.append(self._create_task(
                description=f"Analyze team grit, heart, and intangibles for {self.team1} and {self.team2}. "
                           f"Compare team chemistry, leadership, mental toughness, and clutch performance. "
                           f"Assess intangible factors that could affect the game outcome.",
                expected_output=f"Detailed analysis of intangibles for {self.team1} and {self.team2} "
                               f"including team chemistry, leadership, mental toughness, and clutch performance.",
                agent=agents[task_index]
            ))
            task_index += 1

        # Final prediction task
        prediction_agent = self._create_agent(
            role=f"Expert Game Prediction Analyst for {self.team1} vs {self.team2}",
            goal=f"Analyze all research data and predict the winner and final score for {self.team1} vs {self.team2}",
            backstory=f"You are an expert NFL analyst who specializes in game predictions. "
                     f"Review all analysis data for {self.team1} and {self.team2} and provide a comprehensive prediction "
                     f"including winner, final score, and key factors that will determine the outcome. "
                     f"CRITICAL: Always express your confidence level as a specific percentage (e.g., 75%, 85%, 90%) "
                     f"rather than using vague terms like 'high', 'medium', or 'low'. This provides more precise "
                     f"and actionable information for decision-making."
        )

        prediction_task = self._create_task(
            description=f"Based on all previous analysis, predict the winner and final score for {self.team1} vs {self.team2}. "
                       f"Consider all factors: position groups, coaching, injuries, home field advantage, and intangibles. "
                       f"Provide a detailed explanation of your prediction and confidence level as a PERCENTAGE (e.g., 75%, 85%, 90%). "
                       f"DO NOT use terms like 'high', 'medium', or 'low' - always express confidence as a specific percentage.",
            expected_output=f"Comprehensive game prediction for {self.team1} vs {self.team2} including predicted winner, "
                           f"final score, key factors, and confidence level expressed as a PERCENTAGE (e.g., 75%, 85%, 90%) with detailed reasoning.",
            agent=prediction_agent
        )

        tasks.append(prediction_task)

        return tasks
    
    def process_results(self, results) -> Dict[str, Any]:
        """Process analysis results and create prediction."""
        import re

        # Convert results to string if it's a CrewOutput object
        if hasattr(results, 'raw'):
            results_text = str(results.raw)
        elif hasattr(results, 'output'):
            results_text = str(results.output)
        else:
            results_text = str(results)

        # Extract predicted winner
        predicted_winner = self.team1  # Default
        winner_patterns = [
            r'(?:predicted winner|winner|will win):\s*([^\n,]+)',
            r'([^\n]+)\s+(?:will win|wins|is predicted to win)',
            r'prediction:\s*([^\n]+)\s+(?:wins|will win)',
        ]
        for pattern in winner_patterns:
            match = re.search(pattern, results_text, re.IGNORECASE)
            if match:
                winner_text = match.group(1).strip()
                # Check which team is mentioned
                if self.team1.lower() in winner_text.lower():
                    predicted_winner = self.team1
                    break
                elif self.team2.lower() in winner_text.lower():
                    predicted_winner = self.team2
                    break

        # Extract predicted score
        predicted_score = f"{self.team1} 24, {self.team2} 20"  # Default
        score_patterns = [
            r'(?:final score|predicted score|score):\s*([^\n]+)',
            r'(\d+)\s*-\s*(\d+)',
            r'([A-Za-z\s]+)\s+(\d+)[,\s]+([A-Za-z\s]+)\s+(\d+)',
        ]
        for pattern in score_patterns:
            match = re.search(pattern, results_text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    # Format: 24-20
                    score1, score2 = match.groups()
                    predicted_score = f"{predicted_winner} {score1}, {self.team2 if predicted_winner == self.team1 else self.team1} {score2}"
                    break
                elif len(match.groups()) == 4:
                    # Format: Team1 24, Team2 20
                    team_a, score_a, team_b, score_b = match.groups()
                    predicted_score = f"{team_a.strip()} {score_a}, {team_b.strip()} {score_b}"
                    break
                else:
                    # Use the full match
                    predicted_score = match.group(1).strip()
                    break

        # Extract confidence level
        confidence = '75%'  # Default
        confidence_patterns = [
            r'confidence(?:\s+level)?:\s*([^\n]+)',
            r'(\d+)%\s+confidence',
            r'confidence\s+(?:of\s+)?(\d+)%',
        ]
        for pattern in confidence_patterns:
            match = re.search(pattern, results_text, re.IGNORECASE)
            if match:
                conf_text = match.group(1).strip()
                # Extract percentage if present
                pct_match = re.search(r'(\d+)%', conf_text)
                if pct_match:
                    confidence = f"{pct_match.group(1)}%"
                else:
                    # Check for text-based confidence
                    if 'high' in conf_text.lower():
                        confidence = '85%'
                    elif 'medium' in conf_text.lower() or 'moderate' in conf_text.lower():
                        confidence = '70%'
                    elif 'low' in conf_text.lower():
                        confidence = '60%'
                    else:
                        confidence = conf_text
                break

        # Extract key factors
        key_factors = []
        factors_section = re.search(r'(?:key factors|key matchups|important factors):\s*(.+?)(?:\n\n|\Z)', results_text, re.IGNORECASE | re.DOTALL)
        if factors_section:
            factors_text = factors_section.group(1)
            # Split by bullet points or numbered lists
            factor_lines = re.split(r'[\n•\-\d+\.]', factors_text)
            for line in factor_lines:
                line = line.strip()
                if line and len(line) > 10:  # Ignore very short lines
                    key_factors.append(line)

        # If no factors found, create some based on the analysis
        if not key_factors:
            key_factors = [
                f"Analysis based on 2025 season statistics",
                f"Home field advantage: {self.home_team}",
                f"Comprehensive position group analysis completed",
            ]

        return {
            'team1': self.team1,
            'team2': self.team2,
            'home_team': self.home_team,
            'away_team': self.away_team,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_winner': predicted_winner,
            'predicted_score': predicted_score,
            'confidence': confidence,
            'key_factors': key_factors[:5],  # Limit to 5 factors
            'detailed_analysis': results_text
        }
