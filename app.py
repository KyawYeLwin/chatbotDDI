import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from openai import OpenAI
from langchain.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

app = Flask(__name__)

# LINE Bot configuration
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# OpenAI configuration
openai_apikey = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_apikey)

# Website URLs
website_urls = [
    "https://ddi.au.edu/en/home-en/",
    "https://ddi.au.edu/en/program-information/program-details/",
    "https://ddi.au.edu/en/program-information/study-plan/",
    "https://ddi.au.edu/en/apply-now-2/how-to-apply/",
    "https://ddi.au.edu/en/tuition-fee-2/",
]

# Load and process website data
loader = WebBaseLoader(website_urls)
web_docs = loader.load()

# Load and process PDF files in 'files' directory
pdf_docs = []
pdf_dir = 'files'
for pdf_file in os.listdir(pdf_dir):
    if pdf_file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        pdf_loader = PyMuPDFLoader(pdf_path)
        pdf_docs.extend(pdf_loader.load())

# Combine website and PDF documents
all_docs = web_docs + pdf_docs

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
all_splits = text_splitter.split_documents(all_docs)

# Create embeddings and vector store
openai_embed = OpenAIEmbeddings(openai_api_key=openai_apikey)
convstore = FAISS.from_documents(all_splits, embedding=openai_embed)
retriever = convstore.as_retriever(search_type="similarity")

def openai_function(user_input, additional_context=None):
    system_prompt = f"You are a helpful assistant. Please answer the questions accurately. If you don't know, simply say you don't have enough info. Be positive and energetic in your reply. CONTEXT: {additional_context}. For additional, add a reference or source in your answer."
    response = client.chat.completions.create(
        model='gpt-4-turbo',  # Change the model here to gpt-4-turbo
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        max_tokens=1000,
        temperature=0.8
    )
    return response.choices[0].message.content

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text
    retrieved_docs = retriever.invoke(user_message)
    ai_response = openai_function(user_message, str(retrieved_docs))
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=ai_response)
    )

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
