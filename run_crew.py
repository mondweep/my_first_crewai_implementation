from crewai import Crew, Agent, Task
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Create LangChain chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0.7,
    convert_system_message_to_human=True
)

topic = 'mitosis'
researcher = Agent(
    role=f'{topic} Senior Data Researcher',
    goal=f'Uncover cutting-edge developments in {topic}',
    backstory=f"""You're a seasoned researcher with a knack for uncovering the latest
    developments in {topic}.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

reporting_analyst = Agent(
    role=f'{topic} Reporting Analyst',
    goal=f'Create detailed reports about {topic}',
    backstory=f"""You're a meticulous analyst with a keen eye for detail.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Research Task
research_task = Task(
    description=f"""Conduct a thorough research about {topic}
    Make sure you find any interesting and relevant information about {topic} given
    the current year is 2024.""",
    expected_output=f"""A list with 10 bullet points of the most relevant information about {topic}""",
    agent=researcher
)

# Create and run the crew
crew = Crew(
    agents=[researcher, reporting_analyst],
    tasks=[research_task],
    process="sequential"
)

result = crew.kickoff()
print(result)