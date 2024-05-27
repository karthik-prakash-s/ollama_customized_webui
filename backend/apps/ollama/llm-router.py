import pymilvus
import numpy as np
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Milvus
from langchain.docstore.document import Document
from langchain.utils.math import cosine_similarity
from langchain_core.prompts import PromptTemplate
from typing import List, Union
import re
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from pymilvus import connections, list_collections, Collection

# Load Documents
loader = WebBaseLoader(web_paths=['https://lilianweng.github.io/posts/2023-06-23-agent/'])
docs = loader.load()

# Four Prompts
# Four Prompts
summarize_template = """[Summary Request] You are an expert summarizer. Your task is to provide a concise and accurate summary of the given content.\
Use this prompt only if the query reprents a request for a summary.

Here is the content:
{query}

Please provide the summary of the content."""

direct_question_template = """[Direct Query] You are an expert in answering direct questions like What, Why, When, How, Who, etc.. \
Use this prompt if the query is a straightforward question that can be answered directly.\

Here is the question:
{query}

Please provide a direct answer to the question based on the information provided in the context:
{context}
"""

analytical_question_template = """[Analytical Request] You are an analytical expert skilled in performing  \
mathematical computations and quantitative analysis, provide step by step explanation for your calculations.


Here is the question:
{query}   

Please provide a detailed analysis and evaluation, from the available context.\
If mathematical computations are required, ensure that you clearly present your calculations and explain the reasoning behind them.
{context}"""

reasoning_question_template = """[Reasoning Challenge] You are a knowledgeable assistant with expertise in solving complex reasoning challenges. \
Use this prompt if the query requires logical reasoning, critical thinking, and problem-solving skills to derive an answer. These questions \
often involve multiple steps, deductions or evaluations.
General Example questions:

These examples illustrates the types of reasoning challenges that can be addressed using this prompt template.

Here is the question:
{query}

Please provide a reasoned solution to the challenge based on logical \
analysis and any information provided in the context:
{context}"""

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Step 2: List all collections
all_collections = list_collections()

# Step 3: Drop all collections
for collection_name in all_collections:
    collection = Collection(collection_name)
    collection.drop()
    print(f"Collection '{collection_name}' dropped successfully.")

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

#Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200) 
splits = text_splitter.split_documents(docs)
#print(splits[7])


#Embed
text_documents = [doc.page_content for doc in splits]
documents = [Document(page_content=text) for text in text_documents]
#print(documents)

# Define the ModifiedHuggingFaceEmbeddings class
class ModifiedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts: List[Union[str, dict]]) -> List[List[float]]:
        texts = [re.sub(r'\{[^}]+\}', '', text) if isinstance(text, str) else str(text) for text in texts]
        embeddings = super().embed_documents(texts)
        return embeddings

embedding= ModifiedHuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
# print(embedding)
vectorstore = Milvus.from_documents(documents, embedding, connection_args={"host": "localhost", "port": "19530"})
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
# print(retriever)

model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

#Embed prompts
prompt_templates = [
    summarize_template, 
    direct_question_template, 
    analytical_question_template, 
    reasoning_question_template
]
prompt_embeddings = embedding.embed_documents(prompt_templates) 

similarity_threshold = 0.01  # Adjust this value as needed

# Route question to prompt
def prompt_router(query,context):  
    # Embed question
    query_embedding = embedding.embed_query(query)
    # Compute similarity
    similarities = cosine_similarity([query_embedding], prompt_embeddings)[0]
    similarities = np.absolute(similarities)
    # Check for similarity scores
    if max(similarities) >= similarity_threshold:
        most_similar_index = similarities.argmax()
        most_similar_prompt = prompt_templates[most_similar_index]
        return PromptTemplate.from_template(most_similar_prompt.format(query=str(query), context=str(context)))
    else:
        default_prompt = "I'm sorry, I couldn't determine the appropriate prompt for your query. Please rephrase your question or provide more context."
        return PromptTemplate.from_template(default_prompt)
    

# LLM 
llm_1 = Ollama(model="gemma:2b")
llm_2 = Ollama(model="llama3:latest")

#Post-processing
def format_docs(docs):
    formatted_docs = "\n\n".join(doc.page_content for doc in docs)
    formatted_docs = formatted_docs.replace("{", "").replace("}", "")
    return formatted_docs
                           
                           

rag_chain = (
        RunnableLambda(lambda input_dict: input_dict)
        | RunnableLambda(lambda input_dict: input_dict['prompt_router'])
        | RunnableLambda(lambda input_dict: llm_1 if "[Direct Query]" in str(input_dict) else llm_2)
        | StrOutputParser()
    )


# Handle summary requests separately
def route_to_llm(input_dict):
    prompt_template_str = str(input_dict["prompt_router"])
    if "[Summary Request]" in str(prompt_template_str):
        print("Generating summary directly using LLM...")
        summary_result = llm_1(prompt_template_str.format(query = input_dict["query"]))
        print(summary_result)
    else:
        print("Generate Answer using RAG")
        result = rag_chain.invoke(input_dict)
        print(result)


query = "Give me the summary of the content. Agent System Overview\
In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:\
Planning\
Subgoal and decomposition: The agent breaks down large tasks into smaller, manageable subgoals, enabling efficient handling of complex tasks.\
Reflection and refinement: The agent can do self-criticism and self-reflection over past actions, learn from mistakes and refine them for future steps, \
thereby improving the quality of final results.\
Memory\
Short-term memory: I would consider all the in-context learning (See Prompt Engineering) as utilizing short-term memory of the model \
to learn.\
Long-term memory: This provides the agent with the capability to retain and recall (infinite) information over extended periods, often by leveraging an external vector store and fast retrieval.\
Tool use\
The agent learns to call external APIs for extra information that is missing from the model weights (often hard to change after pre-training), including current information, code execution capability, \
access to proprietary information sources and more."                          

docs = retriever.invoke(query)
print("Documents")
pretty_print_docs(docs)

compressed_docs = compression_retriever.invoke(query)
print('compressed_docs')
pretty_print_docs(compressed_docs)

#prompt_template_str = prompt_router(query,"")
input_dict = {"query": query, "context": format_docs(compressed_docs), "prompt_router": prompt_router(query,format_docs(compressed_docs))}
# prompt_template_str = prompt_router(query, "")
route_to_llm(input_dict)