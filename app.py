from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import prompt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import gradio as gr

GROQ_API_KEY = "groq_api_key"

loader = PyPDFLoader("Bhagavad-Gita.pdf")
docs = loader.load()

text_sp = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_sp.split_documents(docs)

# Extract text content from Document objects
texts = [doc.page_content for doc in splits]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})

prompt_template = """You are an AI trained on Bhagvad Geeta, a sacred Hindu scripture. You provide readings from the text and offer wisdom and guidance based on its teachings.
Your responses should reflect the spiritual and philosophical nature of the Bhagvad Gita, offering deep insights into life's questions.
When asked a question, reference specific verses when appropriate and explain their relevance to the query.
Given below is the context and question of the user,
context = {context}
question = {question}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

vector_store = FAISS.from_texts(texts, embedding=embeddings)
retriever = vector_store.as_retriever()

llm = ChatGroq(model="llama3-8b-8192", 
               groq_api_key=GROQ_API_KEY)

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

def demo(name):
    return rag_chain.invoke(name).content

demo = gr.Interface(fn=demo, inputs="textbox", outputs="textbox", title="Fidem.AI")
demo.launch(share=True)
