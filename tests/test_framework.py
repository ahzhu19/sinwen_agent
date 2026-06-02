from agents.simple_agent import SimpleAgent
from core.llm import BaseLLM
agent = SimpleAgent(name="test", llm=BaseLLM())

result = agent.run("hello")
print(result)