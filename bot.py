from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from data_load.dataLoad import dataLoad

def run_chatbot():
    
    db = dataLoad()
    retriever = db.as_retriever(search_kwargs = {'k':4})

    llm = Ollama(
        model = "llama3.1:latest",
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

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system',system_prompt),    
            ( 'user',"{question}")
        ]
    )

    parser = StrOutputParser()

    # Combine retrieval + prompt + LLM
    chain = (
        {"context": retriever | RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt | llm | parser
    )

    while True:
        question = input('Ask question:')

        if question.lower() == 'bye':
            print('bye')
            break
    
        response = chain.invoke(question)
        print(f'Bot: {response}')
