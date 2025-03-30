import re
import pandas as pd
import streamlit as st
import nest_asyncio
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

nest_asyncio.apply()

# -------------------- Configuration --------------------
# Replace os.getenv(...) with st.secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = st.secrets.get("PINECONE_ENVIRONMENT", "us-east-1-aws")
PINECONE_INDEX_NAME = "thai-patent-index"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GENAI_MODEL = "gemini-2.0-flash"

USER_UPLOADS_DIR = "user_uploads"

# Create directory if it doesn't exist
import os
os.makedirs(USER_UPLOADS_DIR, exist_ok=True)

# -------------------- Page Setup --------------------
st.set_page_config(page_title="Patent Chatbot", layout="wide")
st.title("Chatbot ตรวจสอบสิทธิบัตร (Patent Chatbot) Gemini 2.0 Flash Ver.")

st.markdown(
    """
    <style>
    .user-message {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: right;
        color: black;
    }
    .assistant-message {
        background-color: #F1F0F0;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: left;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- Multipage Setup --------------------
if "page" not in st.session_state:
    st.session_state.page = "main"  # default page is main
if "staff_logged_in" not in st.session_state:
    st.session_state.staff_logged_in = False

# -------------------- Common Functions --------------------
def parse_csv_row(row):
    summary = str(row['บทสรุปการประดิษฐ์']) if pd.notna(row['บทสรุปการประดิษฐ์']) else ""
    claim = str(row['ข้อถือสิทธิ']) if pd.notna(row['ข้อถือสิทธิ']) else ""
    combined_text = summary + " " + claim
    combined_text = ''.join(filter(lambda x: x.isprintable(), combined_text))
    combined_text = ' '.join(combined_text.split())
    return combined_text.strip()

def check_similarity(user_text, embeddings, index):
    user_embedding = embeddings.embed_query(user_text)
    query_response = index.query(vector=user_embedding, top_k=5, include_metadata=True)
    return query_response

def save_user_csv(uploaded_file, save_dir=USER_UPLOADS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def normalize_patent_number(num_str):
    try:
        return str(int(float(num_str)))
    except Exception:
        return num_str

def get_unique_id(row):
    if 'อนุสิทธิบัตร No.' in row and pd.notna(row['อนุสิทธิบัตร No.']) and row['อนุสิทธิบัตร No.'] != "":
        return str(row['อนุสิทธิบัตร No.']).strip()
    elif 'เลขที่คำขอ' in row and pd.notna(row['เลขที่คำขอ']) and row['เลขที่คำขอ'] != "":
        return str(row['เลขที่คำขอ']).strip()
    else:
        return None

# -------------------- Initialize Pinecone & LLM --------------------
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
existing_indexes = pc.list_indexes().names()
index = pc.Index(PINECONE_INDEX_NAME)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
llm = ChatGoogleGenerativeAI(model=GENAI_MODEL, temperature=0.3)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# -------------------- Prompt Templates --------------------
qa_system_prompt = """คุณเป็นระบบช่วยตอบคำถามเกี่ยวกับสิทธิบัตร โดยใช้ข้อมูลที่ได้รับด้านล่างและความรู้ทั่วไปของคุณเพื่อให้คำตอบที่ครอบคลุม:
{context}
หากข้อมูลในฐานข้อมูลไม่เพียงพอหรือมีการเปลี่ยนแปลงคำศัพท์ (เช่น "เจลลี่ขมิ้น" เปลี่ยนเป็น "เจลลี่ขิง") โปรดเสนอแนวทางหรือข้อแนะนำที่เป็นประโยชน์แทนการตอบว่า 'ไม่ทราบ' หรือ 'ยังไม่มีข้อมูล'"""

rag_prompt_template = ChatPromptTemplate.from_template(
    """จงใช้ข้อมูลด้านล่างในการตอบคำถามต่อไปนี้เป็นภาษาไทยเท่านั้น
    หากไม่พบข้อมูลที่เกี่ยวข้อง ให้ตอบว่า 'ไม่ทราบ'
----------
{context}
คำถาม: {question}
คำตอบ:"""
)
history_aware_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question in Thai.
Do NOT answer the question—just reformulate it if needed and otherwise return it as is."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}")
])

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt_template
    | llm
    | StrOutputParser()
)
history_retriever_chain = create_history_aware_retriever(llm, retriever, history_aware_prompt)
qa_chain = create_stuff_documents_chain(llm, qa_prompt)
conversational_rag_chain = create_retrieval_chain(history_retriever_chain, qa_chain)

# -------------------- Define Main Page --------------------
def main_page():
    st.sidebar.button("Staff Login", on_click=lambda: st.session_state.update({"page": "login"}))
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="สวัสดีครับ ยินดีต้อนรับสู่ระบบตรวจสอบข้อถือสิทธิ์ อนุสิทธิบัตรประเภท Food Retail \nของมหาวิทยาลัยเกษตรศาสตร์ หากต้องการให้ตรวจสอบลิขสิทธิ์ กรุณาอัปโหลดไฟล์ของท่านเพื่อตรวจสอบข้อมูล หรือต้องการสอบถามข้อมูลเพิ่มเติมเกี่ยวกับอนุสิทธิบัตรในระบบ สามารถถามเพิ่มเติมได้เลยครับ")]
    
    st.sidebar.header("📂 อัปโหลด Patent Data ของคุณ (Upload Your Patent Data)")
    uploaded_excel = st.sidebar.file_uploader("อัปโหลด Patent Data", type=["xlsx"])
    if uploaded_excel is not None:
        st.session_state.chat_history.append(HumanMessage(content=f"ฉันได้อัปโหลดไฟล์: {uploaded_excel.name}"))
        with st.spinner("กำลังประมวลผล Patent Data..."):
            try:
                df = pd.read_excel(uploaded_excel)
                required_columns = ['บทสรุปการประดิษฐ์', 'ข้อถือสิทธิ']
                if not all(col in df.columns for col in required_columns):
                    st.sidebar.error("Patent Data ของคุณขาดคอลัมน์ที่จำเป็น เช่น 'บทสรุปการประดิษฐ์' หรือ 'ข้อถือสิทธิ'")
                else:
                    df['บทสรุปการประดิษฐ์'] = df['บทสรุปการประดิษฐ์'].fillna('')
                    df['ข้อถือสิทธิ'] = df['ข้อถือสิทธิ'].fillna('')
                    df['combined_text'] = df.apply(parse_csv_row, axis=1)
                    user_text = ' '.join(df['combined_text'].tolist())
                    similarity_results = check_similarity(user_text, embeddings, index)
                    if similarity_results.matches:
                        sim_msg = "**ผลการตรวจสอบความคล้ายของ Patent Data:**\n\n"
                        similar_context = ""
                        for i, match in enumerate(similarity_results.matches, 1):
                            percent_score = match.score * 100
                            sim_msg += f"**Match {i}:**\n"
                            sim_msg += f"- คะแนน: {percent_score:.2f}%\n"
                            sim_msg += f"- ชื่อผลงาน (TH): {match.metadata.get('ชื่อผลงาน (TH)', 'N/A')}\n"
                            sim_msg += f"- ประเภท: {match.metadata.get('ประเภท', 'N/A')}\n"
                            sim_msg += f"- อนุสิทธิบัตร No.: {match.metadata.get('อนุสิทธิบัตร No.', 'N/A')}\n"
                            sim_msg += f"- เจ้าของ: {match.metadata.get('เจ้าของผลงาน', 'N/A')}\n"
                            if match.metadata.get("PDF", "N/A") != "N/A":
                                sim_msg += f"- PDF: [View Patent]({match.metadata.get('PDF')})\n"
                            sim_msg += "\n"
                            similar_context += f"Match {i}: {match.metadata.get('ชื่อผลงาน (TH)', 'N/A')}, {percent_score:.2f}%\n"
                        st.session_state.similar_context = similar_context
                        st.session_state.chat_history.append(AIMessage(content=sim_msg))
                    else:
                        st.session_state.chat_history.append(AIMessage(content="ไม่พบสิทธิบัตรที่คล้ายกันใน Patent Data"))
                    # Do NOT upsert new file data into Pinecone.
            except Exception as e:
                st.sidebar.error(f"เกิดข้อผิดพลาดในการอ่าน Patent Data: {e}")
    if st.sidebar.button("Clear Conversation"):
        st.session_state.chat_history = [AIMessage(content="สวัสดีครับ ยินดีต้อนรับสู่ระบบตรวจสอบข้อถือสิทธิ์ ...")]
    chat_placeholder = st.empty()
    
    def render_chat():
        with chat_placeholder.container():
            for message in st.session_state.chat_history:
                if isinstance(message, HumanMessage):
                    st.markdown(f"<div class='user-message'>{message.content}</div>", unsafe_allow_html=True)
                elif isinstance(message, AIMessage):
                    st.markdown(f"<div class='assistant-message'>{message.content}</div>", unsafe_allow_html=True)
    
    render_chat()
    
    user_input = st.chat_input("พิมพ์คำถามของคุณเกี่ยวกับสิทธิบัตรที่อัปโหลด...")
    if user_input:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.spinner("กำลังประมวลผลคำถามของคุณ..."):
            try:
                response = conversational_rag_chain.invoke({
                    "chat_history": st.session_state.chat_history,
                    "input": user_input
                })
                if isinstance(response, dict):
                    response_text = response.get('text') or response.get('answer') or response.get('response') or str(response)
                elif isinstance(response, str):
                    response_text = response
                else:
                    response_text = 'ไม่ทราบ'
                st.session_state.chat_history.append(AIMessage(content=response_text))
            except Exception as e:
                st.error("ขออภัย, quota ของ API หมด กรุณาลองใหม่อีกครั้งในภายหลัง")
                st.session_state.chat_history.append(AIMessage(content="ขออภัย, ไม่สามารถประมวลผลคำถามได้ในขณะนี้"))
                print(e)
        render_chat()

# -------------------- Define Staff Login and Admin Page --------------------
def login_page():
    st.header("Staff Login")
    staff_username = st.text_input("Username", key="staff_username")
    staff_password = st.text_input("Password", type="password", key="staff_password")
    if st.button("Login"):
        if staff_username == "admin" and staff_password == "password":
            st.success("Logged in successfully!")
            st.session_state.staff_logged_in = True
            st.session_state.page = "admin"
        else:
            st.error("Invalid credentials. Please try again.")
    st.sidebar.button("Return to Main Page", on_click=lambda: st.session_state.update({"page": "main"}))

def admin_page():
    st.header("Admin Page: Update Data from Google Sheet")
    # Add a link to your Google Sheet at the top
    st.markdown("[Go to Google Sheet](https://docs.google.com/spreadsheets/d/1P2v5rCh-jQSZwxGWAO_55DKzNp6kkTuClldsZv1_AMc/edit?gid=0#gid=0)", unsafe_allow_html=True)
    if st.button("Update Data from Google Sheet"):
        def update_from_google_sheet():
            try:
                # Instead of a local JSON file, retrieve from st.secrets
                scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
                service_account_info = st.secrets["google_service_account"]
                creds = ServiceAccountCredentials.from_json_keyfile_dict(service_account_info, scope)
                client = gspread.authorize(creds)

                sheet = client.open("Database อนุสิทธิ Food Business Matching").sheet1
                data = sheet.get_all_records()
                df = pd.DataFrame(data)

                required_columns = ['บทสรุปการประดิษฐ์', 'ข้อถือสิทธิ']
                for col in required_columns:
                    if col in df.columns:
                        df[col] = df[col].fillna('')
                    else:
                        df[col] = ''

                df['combined_text'] = df.apply(parse_csv_row, axis=1)
                documents = []
                metadata_columns = ["ประเภท", "อนุสิทธิบัตร No.", "เลขที่คำขอ", "ชื่อผลงาน (TH)", "ชื่อผลงาน (EN)", "เจ้าของผลงาน", "PDF"]
                
                for idx, row in df.iterrows():
                    unique_id = get_unique_id(row)
                    if unique_id is None:
                        continue
                    metadata = {col: row[col] for col in metadata_columns if col in row}
                    doc = Document(page_content=row['combined_text'], metadata=metadata, id=unique_id)
                    documents.append(doc)

                # Overwrite the existing index: clear all vectors.
                index.delete(delete_all=True)
                # Now add documents to Pinecone
                vector_store.add_documents(documents=documents)
                
                return f"Google Sheet data updated: {len(documents)} documents upserted."
            except Exception as e:
                return f"Error updating from Google Sheet: {e}"
        
        update_message = update_from_google_sheet()
        st.success(update_message)
    st.sidebar.button("Return to Main Page", on_click=lambda: st.session_state.update({"page": "main"}))

# -------------------- Page Navigation --------------------
if st.session_state.get("page") == "login":
    login_page()
elif st.session_state.get("page") == "admin":
    admin_page()
else:
    main_page()
