import unittest
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from loh_langchain import (
    DocumentLoader,
    LangChainVectorDB,
    LangchainModel,
    LohLangChain,
    RedundantChromaFilterRetriever,
)
from _config import OPENAI_API_KEY


embeddings_obj = OpenAIEmbeddings()

chat_model = LangchainModel(model_name="chat_openai")


class TestLangChainAI(unittest.TestCase):

    @unittest.skipIf(True, "Skipping this test if condition is True")
    def test_split_pdf_local(self):
        pdf_file_path = "loh_ai/fhuad_balogun_swe_cv.pdf"
        docs = DocumentLoader().load_pdf_local(pdf_file_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=0,
        )
        split_docs = text_splitter.split_documents(docs)
        print("split_docs:", split_docs[0])
        self.assertIsInstance(split_docs, List)
        self.assertIsInstance(split_docs[0], Document)

    @unittest.skipIf(True, "Skipping this test if condition is True")
    def test_create_embeddings_db(self):
        pdf_file_path = "loh_ai/fhuad_balogun_swe_cv.pdf"
        docs = DocumentLoader().load_pdf_local(pdf_file_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
        split_docs = text_splitter.split_documents(docs)
        lvdb = LangChainVectorDB().create_embeddings_db(
            db_type="chroma",
            docs=split_docs,
            embeddings_obj=embeddings_obj,
        )
        self.assertIsInstance(lvdb, Chroma)

    @unittest.skipIf(False, "Skipping this test if condition is True")
    def test_retrieve(self):
        # query = "which period did I work in Medium Biosciences"
        query = "Give me a summary of my tasks at Medium Biosciences"

        chroma = Chroma(
            persist_directory="chromadb",
            embedding_function=embeddings_obj
        )
        retriever = RedundantChromaFilterRetriever(
            embeddings_obj=embeddings_obj,
            chroma=chroma
        )
        llc = LohLangChain().retrieve(
            query=query,
            retriever=retriever,
            chat_model=chat_model,
        )
        print(llc)
        self.assertIsNotNone(llc)


if __name__ == "__main__":
    unittest.main()
