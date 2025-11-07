from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from nfl_picker.config import CONFIG
from nfl_picker.tools.serper_tool import SerperTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class NflPicker():
    """NflPicker crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    
    # Position Group Agents
    @agent
    def defensive_line_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['defensive_line_researcher'], # type: ignore[index]
            tools=[SerperTool()],
            verbose=True
        )

    @agent
    def linebacker_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['linebacker_researcher'], # type: ignore[index]
            tools=[SerperTool()],
            verbose=True
        )

    @agent
    def cornerback_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['cornerback_researcher'], # type: ignore[index]
            tools=[SerperTool()],
            verbose=True
        )

    @agent
    def safety_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['safety_researcher'], # type: ignore[index]
            tools=[SerperTool()],
            verbose=True
        )

    @agent
    def offensive_lineman_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['offensive_lineman_researcher'], # type: ignore[index]
            tools=[SerperTool()],
            verbose=True
        )

    @agent
    def tight_end_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['tight_end_researcher'], # type: ignore[index]
            tools=[SerperTool()],
            verbose=True
        )

    @agent
    def wide_receiver_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['wide_receiver_researcher'], # type: ignore[index]
            tools=[SerperTool()],
            verbose=True
        )

    @agent
    def running_back_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['running_back_researcher'], # type: ignore[index]
            tools=[SerperTool()],
            verbose=True
        )

    @agent
    def quarterback_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['quarterback_researcher'], # type: ignore[index]
            tools=[SerperTool()],
            verbose=True
        )

    @agent
    def special_teams_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['special_teams_researcher'], # type: ignore[index]
            tools=[SerperTool()],
            verbose=True
        )

    # Team-Wide Analysis Agents
    @agent
    def coaching_staff_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['coaching_staff_researcher'], # type: ignore[index]
            tools=[SerperTool()],
            verbose=True
        )

    @agent
    def team_grit_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['team_grit_researcher'], # type: ignore[index]
            tools=[SerperTool()],
            verbose=True
        )

    @agent
    def home_away_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['home_away_researcher'], # type: ignore[index]
            tools=[SerperTool()],
            verbose=True
        )

    @agent
    def injury_report_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['injury_report_researcher'], # type: ignore[index]
            tools=[SerperTool()],
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    
    # Position Group Research Tasks
    @task
    def defensive_line_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['defensive_line_research_task'], # type: ignore[index]
        )

    @task
    def linebacker_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['linebacker_research_task'], # type: ignore[index]
        )

    @task
    def cornerback_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['cornerback_research_task'], # type: ignore[index]
        )

    @task
    def safety_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['safety_research_task'], # type: ignore[index]
        )

    @task
    def offensive_line_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['offensive_line_research_task'], # type: ignore[index]
        )

    @task
    def tight_end_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['tight_end_research_task'], # type: ignore[index]
        )

    @task
    def wide_receiver_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['wide_receiver_research_task'], # type: ignore[index]
        )

    @task
    def running_back_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['running_back_research_task'], # type: ignore[index]
        )

    @task
    def quarterback_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['quarterback_research_task'], # type: ignore[index]
        )

    @task
    def special_teams_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['special_teams_research_task'], # type: ignore[index]
        )

    # Team-Wide Analysis Tasks
    @task
    def coaching_staff_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['coaching_staff_research_task'], # type: ignore[index]
        )

    @task
    def team_grit_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['team_grit_research_task'], # type: ignore[index]
        )

    @task
    def home_away_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['home_away_research_task'], # type: ignore[index]
        )

    @task
    def injury_report_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['injury_report_research_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the NflPicker crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
