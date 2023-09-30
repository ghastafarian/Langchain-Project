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
model_name = "text-embedding-ada-002"
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
The following information is data from IMDB about certain movies and tv shows, their genres, earnings, ratings, and summaries

{context}

Using this information, answer the following question. The question may ask about specific movies and tv shows, or it may ask about trends.

Question: {question}"""

PROMPT = PromptTemplate(
    template=template, 
    input_variables=["context","question"]
)

chain_type_kwargs = {"prompt": PROMPT}

#method
@cl.on_chat_start
def main():
    #initialize RetrievalQA. This looks through the vectors in the vectorstore, and retrieves the best answer based on the prompt template.
    llm_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs=chain_type_kwargs
    )
    #setting qa as the user session
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["result"]).send()