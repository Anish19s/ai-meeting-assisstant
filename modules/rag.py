from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import time
from google.genai.errors import APIError

def generate_vectorstores(result):

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

    chunks=text_splitter.split_text(result)

    embeddings_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore=FAISS.from_texts(chunks,embeddings_model)

    return vectorstore

def q_and_a(vectorstore,query,client):
   
    docs=vectorstore.similarity_search(query,k=3)

    context="\n".join([doc.page_content for doc in docs])

    prompt=f'''

    You are an AI meeting assisstant

    context:
    {context}


    Question:
    {query}

    Answer clearly and concicesly.
    '''

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                # 3. Upgraded to 2.5-flash for the fastest response and lowest failure rate
                model="gemini-2.5-flash",
                contents=prompt
            )
            break # Success! Break out of the retry loop.

        except APIError as e:
            if e.code in [429, 503] and attempt < 2:
                print(f"Server busy or rate limited ({e.code}). Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"An unexpected error occurred: {e}")
                raise e
            
    return response.text