from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
import fitz  # PyMuPDF for PDF handling
import os
from langchain.chains import GraphCypherQAChain

load_dotenv()
llm = OpenAI()

def main():
    print("LLM test response:", llm.invoke("How are you?"))

    print("📄 Loading PDF...")
    doc = fitz.open("sample_graph_data_extended.pdf")
    text = "\n".join([page.get_text() for page in doc])

    print("✂️ Chunking PDF text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Convert each chunk into a LangChain Document
    documents = [Document(page_content=chunk) for chunk in chunks]

    print("Initializing graph transformer...")
    transformer = LLMGraphTransformer(llm=llm)

    print("Connecting to Neo4j...")
    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="test1234",
        refresh_schema=True 
    )

    print(f"Processing {len(documents)} documents...")
    try:
        graph_docs = transformer.convert_to_graph_documents(documents)
        graph.add_graph_documents(graph_docs)
        print("Graph successfully created in Neo4j!")
    except Exception as e:
        print("Error during graph transformation:", e)

    cypher_prompt = PromptTemplate.from_template(
        """
        You are an expert in converting natural language questions into Cypher queries.
        Assume that all nodes have a `name` property, not an `id`.

        Question: {question}
        """
    )

    chain=GraphCypherQAChain.from_llm(llm=llm,graph=graph,verbose=True,allow_dangerous_requests=True,cypher_prompt=cypher_prompt)
    res=chain.invoke("Who did Neha Sharma work with?")
    print(res)

if __name__ == "__main__":
    main()
