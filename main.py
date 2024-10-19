from flask import Flask, render_template, request, jsonify
from embedchain import App
import os
from replit import db
import csv
import re

# Initialize the Flask app and the embedchain chatbot
app = Flask(__name__)
chatApp = App()

# Add description and instructions for handling messages
instructions = """
Primary Function:  
   You are a legal assistant that answers ethical questions based on the ABA Model Rules of Professional Responsibility, related caselaw, and uploaded materials.

Scope of Responses:  
   You should respond to questions about legal ethics, compile relevant information from documents, summarize it, and cite specific rules, statutes, or caselaw.

Handling Conversations:  
   Start with a friendly greeting, ask how you can assist, and invite users to ask about legal ethics. If they ask off-topic questions, gently guide them back to the appropriate topic.
"""

# Function to check if the user question is related to legal ethics
def related_to_ethics(user_question):
    # Define keywords that indicate the question is about legal ethics
    ethics_keywords = [
        "ethics", "professional responsibility", "ABA rules", "conflict of interest",
        "lawyer duties", "client confidentiality", "ethical responsibilities", "model rule", 
        "competence", "client representation"
    ]
    # Check if the question contains any of these keywords
    return any(keyword in user_question.lower() for keyword in ethics_keywords)

# Function to provide a friendly greeting and ask if the user has any legal ethics questions
def initial_conversation():
    return "Hello! How can I assist you today? Do you have any questions about legal ethics or professional responsibility?"

# Function to continue the conversation and guide the user
def continue_conversation():
    return "Feel free to ask about any ethical dilemmas legal professionals may face, or the ABA Model Rules. I'm here to help!"

# Load PDFs into the chat app and log the process
fileList = os.listdir("docs")
for filename in fileList:
    if filename.endswith(".pdf"):
        keys = db.keys()
        if filename not in keys:
            print(f"Loading {filename} into chatApp")
            chatApp.add("pdf_file", f"docs/{filename}")
            db[filename] = None  # Track that the file has been loaded
        else:
            print(f"{filename} already in db")

# Function to search the PDFs and compile a summary with citations
def search_and_summarize(query):
    print(f"Searching PDFs for query: {query}")

    # Execute the query and capture the response tuple
    try:
        response_tuple = chatApp.query(query)
        response = response_tuple[0]  # Assuming the first element contains the answer

        # Log the entire response for debugging
        print(f"Raw response from embedchain: {response}")

        if not response:
            return "No relevant information found."

        # Extract citations and summarize the content
        citations = extract_citations(response)
        summary = summarize_response(response)

        # Return the combined summary and citations
        return f"Summary: {summary}\n\nCitations: {citations}"

    except Exception as e:
        print(f"Error while searching PDFs: {str(e)}")
        return f"Error while searching documents: {str(e)}"

# Function to extract citations (rules, statutes, caselaw) from the chatbot's response
def extract_citations(response):
    # Example logic to search for patterns of legal citations, e.g., "Model Rule 1.1" or case references
    citation_matches = re.findall(r'Model Rule \d+\.\d+|[A-Z][a-z]+ v\. [A-Z][a-z]+', response)
    return ', '.join(citation_matches) if citation_matches else "No specific citations found."

# Function to summarize the content of the response
def summarize_response(response):
    # Simple logic to generate a summary (this can be made more sophisticated)
    sentences = response.split('. ')
    summary = '. '.join(sentences[:3])  # Grab the first few sentences for summary
    return summary if summary else "No summary available."

# Route for the chatbot page (HTML interface)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle chatbot queries
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')

    try:
        # If the user says 'hello' or starts a conversation, greet them
        if re.search(r'\bhello\b|\bhi\b|\bhey\b', user_message.lower()):
            response_message = initial_conversation()

        # If the message is related to legal ethics, provide the relevant response
        elif related_to_ethics(user_message):
            # Search the PDFs, summarize, and provide citations based on the user's query
            response_message = search_and_summarize(user_message)

        # If no specific legal ethics keywords are found, guide the user back to relevant topics
        else:
            response_message = continue_conversation()

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        response_message = f"Error occurred: {str(e)}"

    return jsonify({'response': response_message})

# Route to handle feedback
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    question = data.get('question')       # The original user question
    response = data.get('response')       # The bot's response
    feedback = data.get('feedback')       # 'helpful' or 'not-helpful'

    # Log the feedback to the console
    print(f"Feedback received for question: '{question}', Response: '{response}', Feedback: '{feedback}'")

    # Save feedback to a CSV file
    feedback_file = 'feedback.csv'
    file_exists = os.path.isfile(feedback_file)

    with open(feedback_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question', 'response', 'feedback']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file does not exist
        if not file_exists:
            writer.writeheader()

        # Write feedback row
        writer.writerow({'question': question, 'response': response, 'feedback': feedback})

    return jsonify({'message': 'Feedback received'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
