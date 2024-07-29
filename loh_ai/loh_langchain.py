from typing import Any, Dict, List
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import S3FileLoader, PyPDFLoader
from langchain_core.documents.base import Document

from loh_ai import AIChainer, BaseVectorDB, LoHModel
from _config import OPENAI_API_KEY


LANGCHAIN_MODELS = {
    "chat_openai": ChatOpenAI()
}


class LangchainModel(LoHModel):
    def __init__(self, model_name: str):
        self.model = LANGCHAIN_MODELS[model_name]


class RedundantChromaFilterRetriever(BaseRetriever):
    embeddings_obj: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query):

        embeddings = self.embeddings_obj.embed_query(query)

        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=embeddings,
            lambda_mult=0.5 # remove very similar documents
        )

    async def aget_relevant_documents(self):
        return []


class DocumentLoader:
    def load_pdf_local(
        self,
        file_path: str,
    ) -> List[Document]:

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents

    def load_file_from_s3(
        self,
        bucket_name: str,
        s3_key: str,
    ) -> List[Document]:

        loader = S3FileLoader(bucket=bucket_name, key=s3_key)
        documents = loader.load()
        return documents


class LangChainVectorDB(BaseVectorDB):

    def create_embeddings_db(
        self,
        db_type: str,
        docs: List[Any],
        embeddings_obj: Any,
    ):
        if db_type == "chroma":
            return self.__create_chroma_db(docs, embeddings_obj)

    def __create_chroma_db(
        self,
        docs: List[Document],
        embeddings_obj: Embeddings = OpenAIEmbeddings,
    ) -> Chroma:

        db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings_obj,
            persist_directory="chromadb",
        )
        return db


class LohLangChain(AIChainer):

    def retrieve(
        self,
        query: str,
        retriever: BaseRetriever = RedundantChromaFilterRetriever,
        chat_model: LoHModel = LangchainModel,
    ):

        chain = RetrievalQA.from_chain_type(
            llm=chat_model.get_model(),
            retriever=retriever,
            chain_type="stuff",
        )
        return chain.run(query)
