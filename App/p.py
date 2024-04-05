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
openai.openai_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Chat with your Documents, powered by LlamaIndex", layout="wide", initial_sidebar_state="auto", menu_items=None)
#st.title("Chat with your Documents")
hide_st_style ="""<style>
                  #MainMenu{visibility:hidden;}
                   footer{visibility:hidden;}
                   header{visibility:hidden;}
                   </style>"""
st.markdown(hide_st_style , unsafe_allow_html=True)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "REGS_AI"}]

# Load data function
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

index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

def query_chatbot(query_engine, user_question):
    response = query_engine.query(user_question)
    return response.response if response else None

def extract_document_section(response_text):
    # Assuming the document section is marked by '[SECTION_START]' and '[SECTION_END]'
    pattern = r'\[SECTION_START\](.*?)\[SECTION_END\]'
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return response_text.strip() 
def generate_response(user_question, chat_engine):
    response = chat_engine.chat(user_question)
    if response:
        response_text = response.response
        document_section = extract_document_section(response_text)
        return response_text, document_section
    return None, None
def display_history():
    st.title("Chat History")
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            role = message['role'].capitalize()
            content = message['content']
            if role == 'User' and 'response' in message:
                st.write(f"{role}: {content}\n\nAssistant: {message['response']}")
            else:
                st.write(f"{role}: {content}")
            if "audio_bytes" in message:
                st.audio(message["audio_bytes"], format='audio/ogg', start_time=0)

def back_to_chat():
    st.session_state.show_chat_history = False

def initialize_chatbot(data_dir="./data", model="gpt-3.5-turbo", temperature=0.3):
    openai.openai_key = st.secrets["OPENAI_API_KEY"]
    documents = SimpleDirectoryReader(data_dir).load_data()
    llm = OpenAI(model=model, temperature=temperature)

    additional_questions_prompt_str = (
        "Given the context below, generate only one additional question different from previous additional questions related to the user's query:\n"
        "Context:\n"
        "User Query: {query_str}\n"
        "Chatbot Response: \n"
    )

    new_context_prompt_str = (
        "We have the opportunity to only one generate additional question different from previous additional questions based on new context.\n"
        "New Context:\n"
        "User Query: {query_str}\n"
        "Chatbot Response: \n"
        "Given the new context, generate only one additional questions different at each time from previous additional questions related to the user's query."
        "If the context isn't useful, generate only one additional questions different at each from previous time from previous additional questions based on the original context.\n"
    )

    chat_text_qa_msgs = [
        (
            "system",
            """Generate only one  additional questions  that facilitate deeper exploration of the main topic 
            discussed in the user's query and the chatbot's response. The questions should be relevant and
              insightful, encouraging further discussion and exploration of the topic. Keep the questions concise 
              and focused on different aspects of the main topic to provide a comprehensive understanding.""",
        ),
        ("user", additional_questions_prompt_str),
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

    # Refine Prompt
    chat_refine_msgs = [
        (
            "system",
            """Based on the user's question '{prompt}' and the chatbot's response '{response}', please 
            generate only one additional questions related to the main topic. The questions should be 
            insightful and encourage further exploration of the main topic, providing a more comprehensive 
            understanding of the subject matter.""",
        ),
        ("user", new_context_prompt_str),
    ]
    refine_template = ChatPromptTemplate.from_messages(chat_refine_msgs)
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        llm=llm,
    )

    return query_engine

def main():
    # Initialize show_chat_history if it doesn't exist
    if "show_chat_history" not in st.session_state:
        st.session_state.show_chat_history = False

    

    if st.sidebar.button("Chat"):
        st.session_state.show_chat_history = False

    if st.sidebar.button("History"):
        st.session_state.show_chat_history = True
        # Clear chat history messages when going back from history
        

    if st.session_state.show_chat_history:
        display_history()  # This will display the chat history if the "History" button is clicked
    else:
        prompt = ""  # Define prompt variable

        if prompt := st.chat_input("Your question"):
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

                    # Save the audio bytes in the session state
                    message["audio_bytes"] = audio_bytes
                    st.session_state.messages.append(message)
                    
                    user_question = prompt
                    additional_questions = query_chatbot(initialize_chatbot(), user_question)
                    #st.write("Additional Questions:")
                    for i, question in enumerate(additional_questions[:1]):   
                        with st.expander(additional_questions, expanded=False):
                            with st.spinner("Thinking..."):
                                answer = chat_engine.chat(question)  # Get the answer for the additional question
                                st.write(f"{i+1}. {answer.response}")
                                #message = {"role": "assistant", "content": additional_questions}
                                #st.session_state.messages.append(message)
                                #additional_question_message = {"role": "assistant", "content": f"{i+1}. {answer.response}"}
                                #st.session_state.messages.append(additional_question_message)

                    user_question = prompt
                    additional_questions = query_chatbot(initialize_chatbot(), user_question)
                    #st.write("Additional Questions:")
                    for i, question in enumerate(additional_questions[:1]):   
                        with st.expander(additional_questions, expanded=False):
                            with st.spinner("Thinking..."):
                                answer = chat_engine.chat(question)  # Get the answer for the additional question
                                st.write(f"{i+1}. {answer.response}")
                                #message = {"role": "assistant", "content": additional_questions}
                                #st.session_state.messages.append(message)
                                #additional_question_message = {"role": "assistant", "content": f"{i+1}. {answer.response}"}
                                #st.session_state.messages.append(additional_question_message)

                    user_question = prompt
                    additional_questions = query_chatbot(initialize_chatbot(), user_question)
                    #st.write("Additional Questions:")
                    for i, question in enumerate(additional_questions[:1]):   
                        with st.expander(additional_questions, expanded=False):
                            with st.spinner("Thinking..."):
                                answer = chat_engine.chat(question)  # Get the answer for the additional question
                                st.write(f"{i+1}. {answer.response}")
                                #message = {"role": "assistant", "content": additional_questions}
                                #st.session_state.messages.append(message)
                                #additional_question_message = {"role": "assistant", "content": f"{i+1}. {answer.response}"}
                                #st.session_state.messages.append(additional_question_message)
                                                        
                    response_text , document_section = generate_response(prompt , chat_engine)
                    if document_section:
                        with st.expander("Reference"):
                            st.write(document_section)
                    else:
                        st.write("No document section found")


if __name__ == "__main__":
    main() 