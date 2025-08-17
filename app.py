from flask import Flask, render_template, jsonify, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.4,
    max_tokens=500
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if 'chat_history' not in session:
        session['chat_history'] = []

    history_from_session = session['chat_history']
    
    chat_history_objects = []
    for message in history_from_session:
        if message['type'] == 'human':
            chat_history_objects.append(HumanMessage(content=message['content']))
        elif message['type'] == 'ai':
            chat_history_objects.append(AIMessage(content=message['content']))

    response = rag_chain.invoke({"input": msg, "chat_history": chat_history_objects})
    
    history_from_session.append({'type': 'human', 'content': msg})
    history_from_session.append({'type': 'ai', 'content': response["answer"]})
    
    session['chat_history'] = history_from_session

    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
