import os
import hashlib
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_community.embeddings import OpenAIEmbeddings
from openai import OpenAI
from pdfminer.high_level import extract_text

# Load environment variables (OpenAI API key)
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

# Initialize the Flask app
app = Flask(__name__)

# Assistant description and instructions (personality injected)
description = """
    You are a friendly and approachable legal assistant with a sharp knowledge of legal ethics. You aim to make complex legal concepts easy to understand and offer a warm, welcoming experience for users.
"""

instructions = """
    You are a legal assistant specializing in legal ethics. You're friendly, professional, and approachable. Feel free to use a casual tone with simple language, and sprinkle in a few emojis to keep the conversation light and engaging when appropriate.

    Respond only to questions related to legal ethics based on the ABA Model Rules and related caselaw. Always provide clear, easy-to-understand answers with a friendly and empathetic tone.

    Feel free to inject a little bit of humor or encouragement when appropriate. Start conversations with a warm greeting, ask how you can help, and wrap up with a friendly note, inviting the user to ask more if needed.

    Remember: You’re professional, but you also want the user to feel comfortable and supported!
"""


# Create OpenAI assistant
assistant = client.beta.assistants.create(
    name="Legal Ethics Assistant",
    description=description,
    instructions=instructions,
    model="gpt-4-turbo-preview",  # Use this model for GPT-4 variant
    tools=[{
        "type": "file_search"
    }],
)

# Debugging: Confirm assistant creation
print(f"Assistant created with ID: {assistant.id}")

# Initialize embeddings (using OpenAI API)
embedding_model = OpenAIEmbeddings()

# Ensure cache directory exists
cache_dir = "embedding_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Function to hash a chunk of text for caching
def hash_text(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# Function to cache and load embeddings to avoid recomputation
def cache_embedding(text_chunk):
    cache_file = f"embedding_cache/{hash_text(text_chunk)}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    embedding = embedding_model.embed_documents([text_chunk])[0]
    with open(cache_file, 'wb') as f:
        pickle.dump(embedding, f)
    return embedding


# Function to extract text from a PDF using pdfminer
def extract_text_from_pdf(pdf_file):
    try:
        return extract_text(pdf_file)
    except Exception as e:
        print(f"Error extracting text from {pdf_file}: {str(e)}")
        return ""

# Process PDF files in the 'docs' directory and extract their text
def process_pdfs_in_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return {}

    pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDFs found in the directory {directory_path}.")
        return {}

    pdf_texts = {}
    for pdf_file in pdf_files:
        full_path = os.path.join(directory_path, pdf_file)
        print(f"Processing {full_path}")
        text = extract_text_from_pdf(full_path)
        if text:
            pdf_texts[pdf_file] = text
        else:
            print(f"Failed to extract text from {pdf_file}")

    return pdf_texts

# Create vector store for legal documents
vector_store = client.beta.vector_stores.create(name="Legal Ethics Documents")
print(f"Vector store ID: {vector_store.id}")

# Assuming your PDF files are located in a folder named 'docs'
file_paths = [os.path.join("docs", filename) for filename in os.listdir("docs") if filename.endswith(".pdf")]

if not file_paths:
    print("No PDF files found.")
else:
    # Upload Policy Files into Vector Store
    file_streams = [open(path, "rb") for path in file_paths]
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )
    # Debugging: Check upload status
    print(file_batch.status)
    print(file_batch.file_counts)

# Chunk text dynamically to avoid processing the entire document at once
def chunk_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Vectorize each chunk using cached embeddings
def vectorize_text_chunks(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        embedding = cache_embedding(chunk)
        embeddings.append(embedding)
    return embeddings

# Create FAISS index for fast retrieval
def create_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

# Retrieve relevant chunks using FAISS index
def retrieve_relevant_chunks(question, index, text_chunks, top_k=5):
    question_embedding = embedding_model.embed_query(question)
    distances, indices = index.search(np.array([question_embedding]).astype('float32'), top_k)
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    return relevant_chunks

# Function to process citations (to extract citations from the assistant's response)
def process_citations(message_content):
    citations = []
    if hasattr(message_content, 'annotations'):
        for index, annotation in enumerate(message_content.annotations):
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = client.files.retrieve(file_citation.file_id)
                citation_text = f"[{index + 1}: {cited_file.filename}]"
                message_content.value = message_content.value.replace(
                    annotation.text, citation_text)
                citations.append(f"{citation_text} {cited_file.filename}")
    return message_content.value, citations

# Preprocess and create embeddings for all the PDFs, and create FAISS index
pdf_texts = process_pdfs_in_directory("docs")  # Process the PDFs
all_text_chunks = []  # Store all text chunks

# Loop through the text and create chunks
for pdf_name, pdf_text in pdf_texts.items():
    chunks = chunk_text(pdf_text)  # Chunk the text
    all_text_chunks.extend(chunks)  # Add the chunks to the list

# Vectorize the text chunks and create FAISS index
if all_text_chunks:
    embeddings = vectorize_text_chunks(all_text_chunks)  # Generate embeddings
    faiss_index = create_faiss_index(embeddings)  # Create FAISS index for search
    
# Flask Route: Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Flask Route: Handle chatbot queries
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')

    if not user_message:
        return jsonify({'response': "Please ask a question related to legal ethics."})

    # Embed user query
    question_embedding = embedding_model.embed_query(user_message)

    # Retrieve relevant chunks using FAISS index
    relevant_chunks = retrieve_relevant_chunks(user_message, faiss_index, all_text_chunks, top_k=5)

    if not relevant_chunks:
        return jsonify({'response': "Sorry, I couldn't find any relevant information."})


    # Combine or summarize the relevant chunks
    assistant_response = " ".join(relevant_chunks)
    
    # Create a thread with the user's message
    thread = client.beta.threads.create(messages=[{
        "role": "user",
        "content": user_message,
    }])

    # Run the assistant and get the response
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # Retrieve the assistant's response
    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
    message_content = messages[0].content[0].text

    # Process citations if necessary
    assistant_response, citations = process_citations(message_content)

    # Initial greeting function with personality
    def initial_conversation():
        return "Hey there! I'm your friendly legal assistant here to help with any legal ethics questions. How can I assist you today?"
    
    # Follow-up question to engage the user further
    follow_up = "Was that helpful? Is there anything else you'd like to know about legal ethics or professional responsibility?"
    
    # Injecting some friendly closing remarks
    closing_note = "If you have any more questions, don't hesitate to ask! I'm here to help 😊"

    # Friendly response with an emoji
    closing_note = "If you need more info, just ask! 😊"

    return jsonify({'response': assistant_response, 'citations': citations})

# Flask Route: Test function to see if PDF content is being searched
@app.route('/test_pdf_query')
def test_pdf_query():
    query = "client confidentiality"  # Adjust this to a query you expect to find in the PDFs
    response = search_and_summarize(query)
    return response

# Flask Route: Handle feedback (Optional)
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    question = data.get('question')
    response = data.get('response')
    feedback = data.get('feedback')

    print(f"Feedback received for question: '{question}', Response: '{response}', Feedback: '{feedback}'")

    feedback_file = 'feedback.csv'
    file_exists = os.path.isfile(feedback_file)

    with open(feedback_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question', 'response', 'feedback']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({'question': question, 'response': response, 'feedback': feedback})

    return jsonify({'message': 'Feedback received'})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
