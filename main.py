import os
from dotenv import load_dotenv
import openai
import PyPDF2
import numpy as np
from pymongo import MongoClient

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

mongdb_username = os.getenv("MONGODB_USERNAME")
mongodb_password = os.getenv("MONGODB_PASSWORD")

mongodb_connection_string = f"mongodb+srv://{mongdb_username}:{mongodb_password}@cluster0.ya56g.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(mongodb_connection_string)
db = client["pdf_rag"]
collection = db["papers"]


# Extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in reader.pages)


# split text into chunks
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # maintain overlap between the chunks
    return chunks


# generate embeddings for text chunks
# def generate_embeddings(text_chunks):
#     embeddings = []
#     for chunk in text_chunks:
#         response = openai.embeddings.create(
#             model="text-embedding-3-small",
#             input=chunk
#         )
#         embeddings.append(response.data[0].embedding)
#     return embeddings


# insert embeddings into MongoDB
# def insert_embeddings_into_mongodb(text_chunks, embeddings):
#     for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
#         document = {
#             "_id": f"chunk-{i}",
#             "text": chunk,
#             "embedding": embedding
#         }
#         collection.insert_one(document)


# generate embeddigns and store it in mongodb
def generate_and_store_embeddings(text_chunks):
    for i, chunk in enumerate(text_chunks):
        # check if the embedding already exists in the database
        existing_doc = collection.find_one({"text": chunk})
        if existing_doc:
            print(f"chunk {i} already exists in the database. Skipping...")
            continue
        # if not, generate and store the embedding
        print(f"Generating embedding for chunk {i}...")
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embedding = response.data[0].embedding

        # insert the chunk and its embedding into MongoDB
        document = {
            "_id": f"chunk-{i}",
            "text": chunk,
            "embedding": embedding
        }
        collection.insert_one(document)


# query MongoDB to retrieve relevant chunks
def query_mongodb(query, top_k=5):
    # generate embedding in query
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding

    # compute similarity and retrieve top_k results
    documents = list(collection.find())
    similarities = []
    for doc in documents:
        embedding = np.array(doc["embedding"])
        similarity = np.dot(query_embedding, embedding) / \
            (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        similarities.append((doc["text"], similarity))

    # sort by similarity and return top_k results
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [text for text, _ in similarities[:top_k]]


# use GPT-4 to generate an answer
def generate_answer(context, query):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant knowledgable in biology, evolution and science."},
            {"role": "user", "content": f"\n{context}\n\nQuestion:{query}\nAnswer:"}
        ],
        temperature=0.2,
        max_tokens=300
    )
    return (response.choices[0].message.content).strip()


# full pipeline
def main(pdf_path, user_query):
    # Extract and process text from PDF...
    print("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)

    print("Splitting text into chunks...")
    text_chunks = split_text(pdf_text)

    # Generate embeddings and store in database
    print("Working on embeddings")
    chunk_embeddings = generate_and_store_embeddings(text_chunks)

    # Inserting embeddings into MongoDB
    # print("Inserting embeddings into MongoDB...")
    # insert_embeddings_into_mongodb(text_chunks, chunk_embeddings)

    print("Querying MongoDB...")
    relevant_chunks = query_mongodb(user_query)
    context = " ".join(relevant_chunks)

    print("Generating answer with GPT-4...")
    answer = generate_answer(context, user_query)

    print("\n Answer")
    print(answer)


if __name__ == "__main__":
    pdf_path = "paper1.pdf"
    user_query = "What is the Darwinian fitness?"
    main(pdf_path, user_query)


# --------DEBUG-------
# text = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."
# chunks = split_text(text)
# print("chunks[0]", chunks[0])
# embeddings = generate_embeddings("I am abhishek sharma.")
# print("embeddings ", embeddings)
