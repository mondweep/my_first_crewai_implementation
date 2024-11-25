from crewai import Agent, Crew, Task
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai.llm import LLM
from langchain.tools import BaseTool, Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
import yaml
import os
from typing import Any
from crewai_tools import SerperDevTool

load_dotenv()

class CustomGoogleLLM(LLM):
    """Custom LLM class to bypass CrewAI's LiteLLM usage"""
    
    def __init__(self):
        # Initialize the underlying LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        # Add required attributes
        self.stop = None  # Required by CrewAI
        self.temperature = 0.7
        self.model = "gemini-pro"
    
    def call(self, prompt, **kwargs):
        """Override the call method to use ChatGoogleGenerativeAI directly"""
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error in CustomGoogleLLM: {e}")
            raise e
            
    @property
    def config(self):
        """Return config dictionary for serialization"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "stop": self.stop
        }

class SearchTool:
    def __init__(self) -> None:
        self._search = GoogleSerperAPIWrapper(
            serper_api_key=os.getenv("SERPER_API_KEY")
        )

    def search(self, query: str) -> str:
        return self._search.run(query)

class Edu:
    """Edu crew"""
    
    def __init__(self):
        # Initialize the LLM
        self.llm = CustomGoogleLLM()
        # Get the research topic from environment variables
        self.topic = os.getenv("CREW_TOPIC", "AI and Machine Learning")  # Default topic if not set
        
        # Create tool instance
        self.tools = [SearchTool()]
        
        # Load configurations from YAML files
        config_dir = os.path.join(os.path.dirname(__file__), 'config')
        
        # Load agents configuration
        with open(os.path.join(config_dir, 'agents.yaml'), 'r') as f:
            self.agents_config = yaml.safe_load(f)
            
        # Load tasks configuration
        with open(os.path.join(config_dir, 'tasks.yaml'), 'r') as f:
            self.tasks_config = yaml.safe_load(f)
        
        self._format_configs_with_topic()

    def _format_configs_with_topic(self):
        """Format all configuration strings with the topic"""
        # Format agents config
        for agent in self.agents_config.values():
            agent['role'] = agent['role'].format(topic=self.topic)
            agent['goal'] = agent['goal'].format(topic=self.topic)
            agent['backstory'] = agent['backstory'].format(topic=self.topic)
        
        # Format tasks config
        for task in self.tasks_config.values():
            task['description'] = task['description'].format(topic=self.topic)
            task['expected_output'] = task['expected_output'].format(topic=self.topic)

    def get_crew(self) -> Crew:
        """Creates the Edu crew"""
        # Create the tool using crewAI's official tool
        search_tool = SerperDevTool()
        
        researcher = Agent(
            role=self.agents_config['researcher']['role'],
            goal=self.agents_config['researcher']['goal'],
            backstory=self.agents_config['researcher']['backstory'],
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
            tools=[search_tool]
        )
        
        analyst = Agent(
            role=self.agents_config['reporting_analyst']['role'],
            goal=self.agents_config['reporting_analyst']['goal'],
            backstory=self.agents_config['reporting_analyst']['backstory'],
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        # Create tasks with assigned agents
        research_task = Task(
            description=f"{self.tasks_config['research_task']['description']} Use the Search tool to find the latest information and developments about {self.topic} from reliable sources.",
            expected_output=self.tasks_config['research_task']['expected_output'],
            agent=researcher
        )
        
        reporting_task = Task(
            description=self.tasks_config['reporting_task']['description'],
            expected_output=self.tasks_config['reporting_task']['expected_output'],
            agent=analyst,
            output_file='report.md'
        )
        
        return Crew(
            agents=[researcher, analyst],
            tasks=[research_task, reporting_task],
            process="sequential",
            verbose=True
        )
