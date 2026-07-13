from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
#from langchain_core.schema import HumanMessage
from langchain_core.messages.human import HumanMessage


load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.environ["GROQ_API_KEY"]
)

# Test the model directly first
messages = [
    HumanMessage(
        content=[
            {
                "type": "text",
                "text": "What do you see in this image?"
            },
            {
                "type": "image_url",
                "image_url": "https://assets.bundesliga.com/contender/2023/11/GettyImages-1268071225.jpg"
            }
        ]
    )
]

response = llm.invoke(messages)
print(response.content)