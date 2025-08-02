# %%
# find and locate .env file in the environment
import os
from dotenv import load_dotenv, find_dotenv
env_file = load_dotenv(find_dotenv())
if not env_file:
    raise FileNotFoundError('.env')
OPEN_API_KEY = os.environ['OPENAI_API_KEY']


# %%
# connect to LLM model
from langchain_openai import OpenAI

llm_model = OpenAI()

# %%
# while True:
response = llm_model.invoke(input("Ask me anything: "))

print(response)

# %%
from langchain_openai import ChatOpenAI
model = 'gpt-3.5-turbo-0125'
chat_model = ChatOpenAI(model=model)

# %%
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ('system', 'digital AI assitant and {title} expert'),
    ('human', '{questions}')
])

messages = prompt.format_messages(
        title='historian',
        questions='tell me about Tipu Sultan'
)

response = chat_model.invoke(messages)

print(response.content)

# %%
# chaining of prompt and llm model
chain = prompt | llm_model

messages = {
    'title': 'historian',
    'questions': 'tell me about Tipu Sultan'
}
response = chain.invoke(messages)

print(response)

# %%
# FewShotPrompting

from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate

examples = [
    
    {'language': 'hindi', 'input': 'hi', 'output': 'स्वागत'},
    {'language': 'hindi', 'input': 'bye', 'output': 'नमस्कार'},
    # {'language': 'hindi', 'input': 'नाम', 'output': 'हिंदी अनुवादक'},
    {'language': 'telugu', 'input': 'hi', 'output': 'హాయ్'},
    {'language': 'telugu', 'input': 'bye', 'output': 'బై'},
]

# example_prompt = ChatPromptTemplate.from_messages(
#     [
#         ('human', '{input}'),
#          ('ai', '{output}')
#     ]
# )


from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate

# the above commented code is acheived similarly from below code using System. Human Message Prompt Template classes
example_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="{output}"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)


few_shot_prompt_template = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'you are an expert {language} translator'),
        few_shot_prompt_template,
        ('human', '{input}')
    ]
)

messages = final_prompt.format_messages(
    language='hindi',
    input='What is your name?'
)

response = chat_model.invoke(messages)

print(response.content)

# %%
chain = final_prompt | chat_model

messages = {
    'language': 'telugu',
    'input': 'can you tell me something about Gandhi family?'
}

response = chain.invoke(messages)

# %%
print(response.content)

# %%
