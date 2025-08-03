# %%
import os
from dotenv import load_dotenv, find_dotenv

env_file = load_dotenv(find_dotenv())

if not env_file:
    raise FileNotFoundError('.env')
    
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# %%
from langchain_openai import OpenAI

llm_model = OpenAI()

# %%
from langchain_openai import ChatOpenAI

model = 'gpt-3.5-turbo-0125'

chat_model = ChatOpenAI(model=model)

# %%
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Tell me a {adjective} story about {topic}!"
)

llm_model_prompt = prompt.format(
    adjective='curious',
    topic='BJP Politics'
)

chat = chat_model.invoke(llm_model_prompt)

# %%
print(chat.content)

# %%
from langchain_core.prompts import ChatPromptTemplate

# help(ChatMessagePromptTemplate)

chat_prompt_msgs = ChatPromptTemplate.from_messages(
    [
        ("system", "you are an expert {language} translator"),
        ("human", "translate anything for to {language}"),
        ("ai", "Sure!"),
        ("human", "{question}")
    ]
)


msgs = chat_prompt_msgs.format_messages(
    language='hindi',
    question='What the BJP tactics of politics?')
    # profession='Political Science',
    # topic='Politics',
    # usr_input='Tell me about the inception of BJP in politics'
# )

resp = chat_model.invoke(msgs)

# %%
resp.content

# %%
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser

json_prompt = PromptTemplate.from_template(
    "Return a json formatted response by attaching an `anwser` key for the following {question}"
)

json_parser = SimpleJsonOutputParser()

json_chain = json_prompt | chat_model | json_parser

# %%
messages = {'question': 'which is the biggest country in the world!'}

resp = json_chain.invoke(messages)

# %%
resp

# %%
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class Politics(BaseModel):
    politics: str = Field(description='question about the political situation')
    country: str = Field(description='answer to specific country\'s political situation')

# %%
parser = JsonOutputParser(pydantic_object=Politics)

# %%
political_prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

custom_output_chain = political_prompt | chat_model | parser

# %%
custom_output_chain.invoke(dict(query='tell me politics of Pakistan'))

# %%
