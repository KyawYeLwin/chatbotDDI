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
    "https://ddi.au.edu/en/contact-for-program-information/",
    "https://ddi.au.edu/en/contact-for-program-information/contact-to-visit/",
    "https://ddi.au.edu/en/ddi-online-learning-2/?lang=en",
    "https://ddi.au.edu/en/category/news-eng/",
    "https://ddi.au.edu/en/apply-now-2/qualifications-of-applicants/",
    "https://ddi.au.edu/en/apply-now-2/application-criteria/",
    "https://ddi.au.edu/en/apply-now-2/required-documents/",
    "https://ddi.au.edu/en/admission-calendar-eng/",
    "https://ddi.au.edu/en/portfolio/youthful-issue-competition/",
    "https://ddi.au.edu/en/portfolio/hackathailand-2023/",
    "https://ddi.au.edu/en/portfolio/young-gen-hackathon-2022/",
    "https://ddi.au.edu/en/program-information/facility/",
    "https://ddi.au.edu/en/program-information/au-spark-2/",
    "https://ddi.au.edu/en/program-information/psytech-assessment-tool-2/",
    "https://ddi.au.edu/en/cooperation-network/",
    "https://ddi.au.edu/en/internship-2/",
    "https://ddi.au.edu/en/partners/guest-speaker-en/",
    "https://ddi.au.edu/en/portfolio/ddi-students-shine-at-startup-competitions/",
    "https://ddi.au.edu/en/program-information/3-years-double-degree-2/",
    "https://ddi.au.edu/en/program-information/ddi-faculty/",
    "https://ddi.au.edu/en/program-information/core-learning/",
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
    system_prompt = f"You are a helpful assistant for DDI (Design and Digital Innovation) program, please answer all the questions the students might have as accurately as you can, specifically about 3rd year programs. If you encounter any questions you are unable to answer then please ask them to contact their professors or department head for more information. Please be very cheerful, energetic and positive with your answers. CONTEXT: {additional_context}. For additional, add a reference or source in your answer."
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