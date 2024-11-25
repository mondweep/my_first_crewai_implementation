from langchain_community.chat_models import ChatLiteLLM
from langchain.schema import HumanMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# def test_gemini_endpoint():
#     try:
#         # Initialize the LLM with the Gemini model
#         llm = ChatLiteLLM(
#             model="gemini/gemini-1.5-pro",
#             api_key=os.getenv("GEMINI_API_KEY"),
#             temperature=0.7
#         )

#         # Create a chat message
#         messages = [HumanMessage(content="Hello, how are you?")]
        
#         # Make a test call to the LLM
#         response = llm.generate([messages])
#         print("Type of response:", type(response))
#         print("Response:", response)

#     except Exception as e:
#         print("Error occurred:", e)

def test_search_endpoint():
    try:
        # Initialize the Serper API wrapper
        search = GoogleSerperAPIWrapper(
            serper_api_key=os.getenv("SERPER_API_KEY")
        )

        # Make a test search query
        query = "Latest developments in AI 2024"
        response = search.run(query)
        
        print("Search Query:", query)
        print("\nSearch Results:", response)

    except Exception as e:
        print("Error occurred:", e)

if __name__ == "__main__":
    # test_gemini_endpoint()
    test_search_endpoint()
