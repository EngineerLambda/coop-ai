import os
from dotenv import dotenv_values
from flask import Flask, jsonify, request, render_template
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# Loading secrets and setting up as environment variables
env_vars = dict(dotenv_values())
os.environ["GOOGLE_API_KEY"] = env_vars["gemini-apikey"]
os.environ["PINECONE_API_KEY"] = env_vars["pinecone-apikey"]


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'

@app.route("/")
def home():
    return render_template("index.html")


def load_doc(file_name):
    try: 
        # Load the pdf document content
        loader = PyPDFLoader(file_name)
        doc = loader.load()
        
        # Split the document
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(doc)
        
        # use google embeddings
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Pushing to vectorstore for docsearch and retrieval purpose
        doc_search = PineconeVectorStore.from_documents(docs, embedding=embedding, index_name="coop-test") # doc_search can be used later on
        
        # message response
        message = {"message" : "Successfully uploaded to vector store"}
        return render_template("success.html"), 200
    
    except Exception as e:
        error_msg = {"error" : f"Error encountered: {e}"}
        return jsonify(error_msg), 400


@app.route("/upload_doc", methods=["GET", "POST"])
def upload_docs():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_name = file.filename
            *_, ext = file_name.split(".")
            
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
            
            # call function to laod doc to vectorDB
            response = load_doc(file_name="uploads/" + file_name)
            return response
        
    response = {"message": "No file uploaded"}
    return jsonify(response)

 
if __name__ == "__main__":
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.run(port=5000, debug=True)
