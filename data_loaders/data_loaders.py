# %%
import os
from dotenv import load_dotenv, find_dotenv

env_file = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# %%
from langchain_openai import ChatOpenAI

model = 'gpt-3.5-turbo-0125'

chat_model = ChatOpenAI(model=model)

# %%
from langchain_community.document_loaders import TextLoader

loader = TextLoader('dataset/be-good.txt')

# %%
loaded_data = loader.load()

# %%
# loaded_data[0].page_content

# %%
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_template(
    "Can you suggest me a title to this dataset: {dataset}?"
)

message = {
    'dataset': loaded_data[0].page_content
}

message = chat_prompt.format_messages(
    dataset=loaded_data
)

resp = chat_model.invoke(message)

# %%
resp.content

# %%
"""
# RAG Method of handling large datasets/volumes
"""

# %%
"""
# _CharacterTextSplitter_ Technique
"""

# %%
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator='\n\n',
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False
)

# %%
text = splitter.create_documents(
    [loaded_data[0].page_content]
)

# %%
len(text)

# %%
metadata = [{'chunk': 0}, {'chunk': 1}]

text_with_metadata = splitter.create_documents(
    [
        loaded_data[0].page_content, loaded_data[0].page_content
    ],
    metadatas=metadata
)

# %%
text_with_metadata[0]

# %%
"""
# _RecursiveSplitter_ Technique
"""

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_overlap=4,
    chunk_size=26
)

# %%
text1 = 'abcdefghijklmnopqrstuvwxyz'

# %%
text2 = """
Data that Speak
LLM Applications are revolutionizing industries such as 
banking, healthcare, insurance, education, legal, tourism, 
construction, logistics, marketing, sales, customer service, 
and even public administration.

The aim of our programs is for students to learn how to 
create LLM Applications in the context of a business, 
which presents a set of challenges that are important 
to consider in advance.
"""

# %%
recursive_splitter.split_text(text1)

# %%
recursive_splitter.split_text(text2)

# %%
second_recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_overlap=0,
    chunk_size=150,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)

# %%
