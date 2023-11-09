
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import datetime



load_dotenv()
# Create a function to save chat history to a file
def save_chat_history(messages):
    with open("chat_history.pkl", "wb") as file:
        pickle.dump(messages, file)

# Create a function to load chat history from a file
def load_chat_history():
    if os.path.exists("chat_history.pkl"):
        with open("chat_history.pkl", "rb") as file:
            return pickle.load(file)
    #else:
        #return []

# Initialize session state with loaded chat history or an empty list
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()


# "New Chat" button
if st.button("New Chat", key="start_new_chat", help="Click to start a new chat"):
    st.session_state.is_chatting = True
    st.session_state.messages = []

with st.sidebar:
    st.title('PDF Chat App')

    # Display the chat history in the sidebar
    st.write("Chat History:")
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            with st.expander(f"User ({message['timestamp']})"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.expander(f"Assistant ({message['timestamp']})"):
                st.markdown(message["content"])

    add_vertical_space(5)
    st.write('Made by Preetham G')

# Rest of your Streamlit app code remains the same

def main():
    st.header("Chat with PDF 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        query = st.text_input("Ask questions about your PDF file:")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        if query:
            chat_history = []
            with st.chat_message("user"):
                st.markdown(query)
            user_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.messages.append({"role": "user", "content": query, "timestamp": user_timestamp})

            custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. At the end of standalone question add this 'Answer the question in English language.' If you do not know the answer reply with 'I am sorry'.
                                   Chat History:
                                   {chat_history}
                                   Follow Up Input: {question}
                                   Standalone question:
                                   Remember to greet the user with hi welcome to pdf chatbot how can i help you? if user asks hi or hello """
            CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
            llm = ChatOpenAI()
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm,
                VectorStore.as_retriever(),
                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                memory=memory
            )
            response = conversation_chain({"question": query, "chat_history": chat_history})
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
            assistant_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.messages.append({"role": "assistant", "content": response["answer"], "timestamp": assistant_timestamp})
            chat_history.append((query, response))

#if st.button("Refresh"):
    #st.session_state.messages = []

if __name__ == '__main__':
    main()

# Save chat history to a file when the app is closed
st.write("Saving chat history to file...")
save_chat_history(st.session_state.messages)
