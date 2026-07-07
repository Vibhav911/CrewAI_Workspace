# CrewAI 1.15.1 adds a cache_breakpoint marker that Groq rejects.
# Disable that marker for this notebook so Groq requests stay compatible.
from crewai.llms import cache as crew_cache


def _patched_mark_cache_breakpoint(message):
    return message


crew_cache.mark_cache_breakpoint = _patched_mark_cache_breakpoint



# Importing libraries
from crewai import Agent, Task, LLM, Crew
from crewai_tools import SerperDevTool

from dotenv import load_dotenv
import os


# Loading Environment Variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")


# A1 Researcher + Web Search (Serper API) --->  A2 Content Writer (Summarization)

# LLm Provider
llm = LLM(
     model="groq/llama-3.3-70b-versatile",
     temperature = 0.7
)


topic = "AI in Healthcare"

# Tool
search_tool = SerperDevTool(n=2)


# Agent 1
senior_research_analyst = Agent(
     name = "Senior Research Analyst",
     role = "Senior Research Analyst",
     goal = f"Research, analyse and synthesize comprehensive information on {topic} from reliable web sources.",
     backstory="You're an expert research analyst with advanced web research skills. "   # Backstory should be defined well
               "You excel at finding, analyzing, and synthesizing information from "     # Write prompt in detail to get 
               "across the internet using search tools. You're skilled at "              # better response
               "distinguishing reliable sources from unreliable ones, "                  # Be very descriptive 
               "fact-checking, cross-referencing information, and " 
               "identifying key patterns and insights. You provide " 
               "well-organized research briefs with proper citations "
               "and source verification. Your analysis includes both " 
               "raw data and interpreted insights, making complex " 
               "information accessible and actionable.",
     verbose = True,
     allow_delegation=False,  # No interconnection. Agent will info to other agent sequentially. No both way communication
     tools = [search_tool],
     llm = llm
)


# Content Writer
content_writer = Agent(
     name = "Content Writer",
     role="Content Writer",
     goal="Transform research findings into engaging blog posts while maintaining accuracy.",
     backstory="You're a skilled content writer specialized in creating "
               "engaging, accessible content from technical research. "
               "You work closely with the Senior Research Analyst and excel at maintaining the perfect "
               "balance between informative and entertaining writing, "
               "while ensuring all facts and citations from the research "
               "are properly incorporated. You have a talent for making "
               "complex topics approachable without oversimplifying them.",
     verbose=True,
     allow_delegation=False,
     llm=llm
)

# Research Task
research_task = Task(
     name = "Research Task",
     description="""   
            1. Conduct comprehensive research on {topic} including:
                - Recent developments and news
                - Key industry trends and innovations
                - Expert opinions and analyses
                - Statistical data and market insights
            2. Evaluate source credibility and fact-check all information
            3. Organize findings into a structured research brief
            4. Include all relevant citations and sources
     """,
    expected_output = """A detailed research report containing:
            - Executive summary of key findings
            - Comprehensive analysis of current trends and developments
            - List of verified facts and statistics
            - All citations and links to original sources
            - Clear categorization of main themes and patterns
            Please format with clear sections and bullet points for easy reference.""",
    agent = senior_research_analyst
)

# Content Writer Task
writing_task = Task(
     name = "Writing Task",
     description="""
          Using the research brief provided, create an engaging blog post that:
          1. Transforms technical information into accessible content
          2. Maintains all factual accuracy and citations from the research
          3. Includes:
               - Attention-grabbing introduction
               - Well-structured body sections with clear headings
               - Compelling conclusion
          4. Preserves all source citations in [Source: URL] format
          5. Includes a References section at the end
     """,
     expected_output = """A polished blog post in markdown format that:
          - Engages readers while maintaining accuracy
          - Contains properly structured sections
          - Includes Inline citations hyperlinked to the original source url
          - Presents information in an accessible yet informative way
          - Follows proper markdown formatting, use H1 for the title and H3 for the sub-sections""",
     agent = content_writer
)


crew = Crew(
     agents = [senior_research_analyst, content_writer],
     tasks = [research_task, writing_task],
     verbose = True
)

result = crew.kickoff(
     inputs={
          "topic": topic
     }
)

print(result)