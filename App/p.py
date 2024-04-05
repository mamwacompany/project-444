import asyncio
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from gtts import gTTS
import io
from llama_index.core.prompts.base import ChatPromptTemplate
import json
import openai
import re
import pyttsx3
import os

# Initialize the event loop
loop = asyncio.get_event_loop()

# Set OpenAI API key
openai.openai_key = st.secrets["OPENAI_API_KEY"]

# Function to load and index data
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Documents...."):
        data_dir = os.path.abspath(os.path.join(os.path.dirname("data"), "data"))
        reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
        docs = reader.load_data()
        embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        llm = OpenAI(model="gpt-3.5-turbo", temperature="0.1", systemprompt="""Use the books in data file is source for the answer.Generate a valid 
                     and relevant answer to a query related to 
                     construction problems, ensure the answer is based strictly on the content of 
                     the book and not influenced by other sources. Do not hallucinate. The answer should 
                     be informative and fact-based. """)
        service_content = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
        index = VectorStoreIndex.from_documents(docs, service_context=service_content)
        return index

# Load data
index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Main function
def main():
    # Set Streamlit page configuration
    st.set_page_config(page_title="Chat with your Documents, powered by LlamaIndex", layout="wide", initial_sidebar_state="auto", menu_items=None)
    hide_st_style ="""<style>
                      #MainMenu{visibility:hidden;}
                       footer{visibility:hidden;}
                       header{visibility:hidden;}
                       </style>"""
    st.markdown(hide_st_style , unsafe_allow_html=True)

    # Initialize show_chat_history if it doesn't exist
    if "show_chat_history" not in st.session_state:
        st.session_state.show_chat_history = False

    if st.sidebar.button("Chat"):
        st.session_state.show_chat_history = False

    if st.sidebar.button("History"):
        st.session_state.show_chat_history = True

    if st.session_state.show_chat_history:
        display_history()  # This will display the chat history if the "History" button is clicked
    else:
        prompt = st.chat_input("Your question")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "audio_bytes" in message:
                    st.audio(message["audio_bytes"], format='audio/ogg', start_time=0)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_engine.chat(prompt)
                    st.write(response.response)
                    message = {"role": "assistant", "content": response.response}

                    # Add a button to play the audio
                    tts = gTTS(response.response)
                    audio_bytes_io = io.BytesIO()
                    tts.write_to_fp(audio_bytes_io)
                    audio_bytes_io.seek(0)
                    audio_bytes = audio_bytes_io.read()
                    st.audio(audio_bytes, format='audio/ogg', start_time=0)

                    message["audio_bytes"] = audio_bytes
                    st.session_state.messages.append(message)
                    
                    user_question = prompt
                    additional_questions = query_chatbot(initialize_chatbot(), user_question)
                    for i, question in enumerate(additional_questions[:1]):   
                        with st.expander(additional_questions, expanded=False):
                            with st.spinner("Thinking..."):
                                answer = chat_engine.chat(question)  
                                st.write(f"{i+1}. {answer.response}")

                    response_text , document_section = generate_response(prompt , chat_engine)
                    if document_section:
                        with st.expander("Reference"):
                            st.write(document_section)
                    else:
                        st.write("No document section found")

if __name__ == "__main__":
    main()
