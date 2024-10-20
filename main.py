import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain.text_splitter import CharacterTextSplitter
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

# Function to chunk text
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        separator=" ", 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Add chunks to the vector store
def add_chunks_to_vector_store(chunks, vector_store_id):
    for i, chunk in enumerate(chunks):
        document_id = f"chunk_{i}"
        client.beta.vector_stores.documents.add(
            vector_store_id=vector_store_id,
            documents=[{"text": chunk, "document_id": document_id}]
        )
    print(f"Added {len(chunks)} chunks to the vector store")

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

# Flask Route: Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Flask Route: Handle chatbot queries
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')

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
