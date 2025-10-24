from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import pickle
import os
import PyPDF2
from collections import deque
from dotenv import load_dotenv

def env(key: str, default=None):
    return os.getenv(key, default)

COHERE_API_KEY = env("COHERE_API_KEY")
MEMORY_FILE = "chatbot_memory.pkl"
PDF_FILE = "sotaysinhvien2025.pdf"

# Đọc nội dung PDF
def load_pdf_content(pdf_path):
    """Trích xuất nội dung từ file PDF"""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        print(f"✓ Đã tải PDF: {pdf_path}\n")
    except Exception as e:
        print(f"✗ Lỗi khi đọc PDF: {e}\n")
    return text

# Custom chat history class để lưu vào pickle
class PersistentChatHistory(BaseChatMessageHistory):
    def __init__(self, messages=None):
        self.messages = list(messages) if messages else []
    
    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
    
    def clear(self) -> None:
        self.messages = []

# Tải nội dung PDF
pdf_content = load_pdf_content(PDF_FILE) if os.path.exists(PDF_FILE) else ""

# Khởi tạo LLM
llm = ChatCohere(model="command-a-03-2025", cohere_api_key=COHERE_API_KEY)

# Tải hoặc tạo mới chat history
if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "rb") as f:
            chat_history = pickle.load(f)
        # Kiểm tra xem có phải PersistentChatHistory không
        if not isinstance(chat_history, PersistentChatHistory):
            chat_history = PersistentChatHistory()
        print("📂 Tải lịch sử hội thoại cũ\n")
    except:
        chat_history = PersistentChatHistory()
        print("✨ Bắt đầu cuộc trò chuyện mới\n")
else:
    chat_history = PersistentChatHistory()
    print("✨ Bắt đầu cuộc trò chuyện mới\n")

# Tạo prompt template với nội dung PDF
# Escape curly braces để tránh lỗi template variable
escaped_pdf = pdf_content.replace("{", "{{").replace("}", "}}")
system_message = f"""Bạn là một trợ lý hữu ích. Sử dụng thông tin từ tài liệu sau để trả lời câu hỏi:

{escaped_pdf}

Nếu bạn không tìm thấy thông tin cần thiết trong tài liệu, hãy cho biết rõ điều đó."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

print("🤖 Chatbot Console")
print("Gõ 'quit' hoặc 'exit' để thoát\n")

try:
    while True:
        user_input = input("Bạn: ").strip()
        
        if user_input.lower() in ["quit", "exit"]:
            # Lưu lịch sử trước khi thoát
            with open(MEMORY_FILE, "wb") as f:
                pickle.dump(chat_history, f)
            print("💾 Đã lưu lịch sử hội thoại")
            print("Tạm biệt! 👋")
            break
        
        if not user_input:
            continue
        
        # Gọi chain với input và history
        user_message = HumanMessage(content=user_input)
        chat_history.add_message(user_message)
        
        response = chain.invoke({
            "input": user_input,
            "history": chat_history.messages
        })
        
        ai_message = AIMessage(content=response.content)
        chat_history.add_message(ai_message)
        print(f"Bot: {response.content}\n")
except KeyboardInterrupt:
    # Lưu lịch sử nếu bị interrupt (Ctrl+C)
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(chat_history, f)
    print("\n💾 Đã lưu lịch sử hội thoại")
    print("Tạm biệt! 👋")
