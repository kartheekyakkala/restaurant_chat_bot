import logging
from chromadb.config import Settings
import click
import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)
from langchain import HuggingFaceInstructEmbeddings
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY




model_id="impira/layoutlm-document-qa"
with open ("./constitution.pdf") as f:
    documents = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100,length_function = len,add_start_index = True)
texts = text_splitter.create_documents(documents)

embeddings = HuggingFaceInstructEmbeddings(
    model_name="microsoft/layoutlm-base-uncased",
    model_kwargs={"device": "cpu"},  # Specify the device type (e.g., "cpu" or "cuda")
)
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory="./DB",
        client_settings=CHROMA_SETTINGS,
    )
db.persist()
# db = Chroma(
#         persist_directory=PERSIST_DIRECTORY,
#         embedding_function=embeddings,
#         client_settings=CHROMA_SETTINGS,
#     )
retriever = db.as_retriever()
# docsearch = Chroma.from_documents(texts, embeddings)
# tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-document-qa")
# model = AutoModelForDocumentQuestionAnswering.from_pretrained("impira/layoutlm-document-qa")
# Use a pipeline as a high-level helper


llm = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
while True:
    query = input("\nEnter a query: ")
    if query == "exit":
        break
    # Get the answer from the chain
    res = qa(query)
    answer, docs = res["result"], res["source_documents"]

    # Print the result
    print("\n\n> Question:")
    print(query)
    print("\n> Answer:")
    print(answer)

    # if show_sources:  # this is a flag that you can set to disable showing answers.
        # # Print the relevant sources used for the answer
    print("----------------------------------SOURCE DOCUMENTS---------------------------")
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)
    print("----------------------------------SOURCE DOCUMENTS---------------------------")






