import os
import pinecone
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import chainlit as cl
llm = OpenAI(temperature=0)
#initialize pinecone object. The keys let me utilize my specific database
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),  
    environment=os.getenv('PINECONE_ENV')  
)
#initialize embeddings
model_name = "gpt-3.5-turbo"
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)
#initialize vectorstore. This turns the contents of the "langchain-quickstart" index in my database into vectors to call from.
index_name = "langchain-quickstart"
index = pinecone.Index(index_name)
text_field = "text"
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

#intialize prompt. This defines the context of the question being asked, to help narrow down the correct answer.
template= """
You are an expert data scientist with an expertise in building deep learning models.

{context}

Question: {question}"""

PROMPT = PromptTemplate(
    template=template, 
    input_variables=["context","question"]
)

chain_type_kwargs = {"prompt": PROMPT}

#method
@cl.langchain_factory(use_async=True)
def main():
    #initialize RetrievalQA. This looks through the vectors in the vectorstore, and retrieves the best answer based on the prompt template.
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs=chain_type_kwargs
    )
    #running qa to ask a question
    return qa

