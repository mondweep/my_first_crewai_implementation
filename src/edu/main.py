#!/usr/bin/env python
import sys
import warnings
import os
from dotenv import load_dotenv

from edu.crew import Edu

## Must precede any llm module imports

from langtrace_python_sdk import langtrace
from crewai import Agent, Crew, Task
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
langtrace.init(api_key=os.getenv('LANGTRACE_API_KEY'))

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    print(f"Current topic from env: {os.getenv('CREW_TOPIC')}")
    
    load_dotenv(override=True)
    print(f"Topic after reload: {os.getenv('CREW_TOPIC')}")
    
    edu = Edu()
    print("I have instantiated the Edu class")
    crew = edu.get_crew()
    print("I have instantiated the crew")
    print(crew)
    result = crew.kickoff()
    print("I have kicked off the crew")
    print(result)

if __name__ == "__main__":
    run()

def train():
    """
    Train the crew for a given number of iterations.
    """
    topic = os.getenv('CREW_TOPIC', 'AI LLMs')  # Default topic if none provided
    inputs = {
        "topic": topic
    }
    try:
        print("I am training the crew")
        Edu().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
        print("I have trained the crew")

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Edu().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    topic = os.getenv('CREW_TOPIC', 'AI LLMs')  # Default topic if none provided
    inputs = {
        "topic": topic
    }
    try:
        Edu().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
