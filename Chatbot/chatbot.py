from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import openai
import os

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

reader = PdfReader("../Document/React.pdf")

raw_text = ''

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len,
)

texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()

docsearch = faiss.FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(openai.OpenAI(), chain_type="stuff")

query = input("Ask your Question : \n")
docs = docsearch.similarity_search(query)
ans = chain.run(input_documents=docs, question=query)
print(ans)

