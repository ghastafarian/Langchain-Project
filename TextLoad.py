from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv,find_dotenv
from tqdm.autonotebook import tqdm
load_dotenv(find_dotenv())
import os
import pinecone
from langchain.vectorstores import Pinecone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),  
    environment=os.getenv('PINECONE_ENV')  
)

with open('myfavs.txt') as f:
    contents = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=0)
texts = text_splitter.create_documents([contents])
embed = OpenAIEmbeddings()
query_result = embed.embed_query(texts[0].page_content)
index_name = "langchain-quickstart"
index = pinecone.Index(index_name)
delete_response = index.delete(delete_all=True)
search = Pinecone.from_documents(texts, embed, index_name=index_name)