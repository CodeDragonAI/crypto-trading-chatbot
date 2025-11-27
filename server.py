from fastapi import FastAPI
from pydantic import BaseModel
from data_load.dataLoad import dataLoad

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = FastAPI()

db = dataLoad()
retriever = db.as_retriever(search_kwargs = {'k': 4})

llm = Ollama(
    model = 'llama3.1:latest',
    temperature = 0.7
)

system_prompt = (
        """You are a confident and helpful trading assistant. 
            Use the following context to answer naturally and clearly, as if you're explaining from your own expertise. 
            **Do not mention documents, sources, or say things like 'from what I understand', 'based on the documents', or 'it seems'.** 
            Just explain the concepts directly and confidently, like a mentor teaching a smart beginner.

            Context:
            {context}
            """
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

chain = (
    {'context': retriever | RunnablePassthrough(), 'question': RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

class Query (BaseModel):
    question: str

@app.post('/ask')
def ask_question(query: Query):
    answer  = chain.invoke(query.question)
    return {'answer': answer}

@app.get('/')
def home():
    return{'message': 'Trading Chatbot API is running'}