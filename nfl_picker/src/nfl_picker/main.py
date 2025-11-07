#!/usr/bin/env python
import sys
import warnings

# Apply SSL fix BEFORE any other imports
from nfl_picker.ssl_fix import apply_ssl_fix
apply_ssl_fix()

from nfl_picker.crew import NflPicker
from nfl_picker.utils import create_analysis_inputs, retry_on_network_error

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


@retry_on_network_error(max_retries=3, initial_delay=5)
def run():
    """
    Run the crew.
    """
    inputs = create_analysis_inputs()

    print("Attempting to run crew...")
    NflPicker().crew().kickoff(inputs=inputs)
    print("Crew execution completed successfully!")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = create_analysis_inputs()
    try:
        NflPicker().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        NflPicker().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = create_analysis_inputs()
    try:
        NflPicker().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
