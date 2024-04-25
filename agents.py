import shelve

import pinecone
from flask import Flask, render_template, request, Response, stream_with_context
from flask_cors import CORS  # Import CORS
import webbrowser
from threading import Timer
import os
from getpass import getpass

from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from pinecone import ServerlessSpec
import time
from langchain_community.vectorstores import Pinecone
import os
from getpass import getpass
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor
from langchain.agents import initialize_agent

from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

from langchain_community.callbacks import StreamlitCallbackHandler
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
class CallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.content: str = ""
        self.final_answer: bool = False

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.content += token
        if "Final Answer" in self.content:
            # now we're in the final answer section, but don't print yet
            self.final_answer = True
            self.content = ""
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ["}"]:
                    sys.stdout.write(token)  # equal to `print(token, end="")`
                    sys.stdout.flush()

load_dotenv('api.env')
# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_API_KEY")

# configure client
pc = pinecone.Pinecone(api_key='6c7958da-3550-466b-ab92-4bf9e5c6e136')

# get API key from top-right dropdown on OpenAI website
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key='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'
)

# connect to index
index = pc.Index("langchain-retrieval-agent")


text_field = "text"  # the metadata field that contains our text

# initialize the vector store object
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# chat completion llm
llm = ChatOpenAI(
    openai_api_key='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',
    model_name='gpt-3.5-turbo',
    temperature=0.0,
    streaming=True,  # ! important
    callbacks=[CallbackHandler()]  # ! important
)
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

tools = [
    Tool(
        name='Knowledge Base HSE',
        func=qa.run,
        description=(
            'use this tool when answering general knowledge queries to get '
            'more information about the HSE University'
        )
    )
]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=4,
    early_stopping_method='generate',
    memory=conversational_memory,
    return_intermediate_steps=False
)


# Simulated Agent Class - Replace this with your actual agent class

# def streamlit_f():
#
#     st.title('Higher School of Economics Assistant')
#     # Display chat messages from history on app rerun
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#
#     if prompt := st.chat_input():
#         # st.chat_message("user").write(prompt)
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         # Display user message in chat message container
#         with st.chat_message("user"):
#             st.markdown(prompt)
#         with st.chat_message("assistant"):
#             st.write("thinking...")
#             #st_callback = StreamlitCallbackHandler(st.container())
#             response = agent.invoke({"input": prompt})
#             #chat_history.append(HumanMessage(content=prompt))
#             st.session_state.messages.append({"role": "assistant", "content": response['output']})
#             st.markdown(response)
#
# if __name__ == "__main__":
#     streamlit_f()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = agent.invoke({"input": user_message})
    print(response)
    # buffer = []
    # buffer_size = 3
    #
    # def generate():
    #     for token in response['output']:
    #         buffer.append(token)
    #         if len(buffer) >= buffer_size:
    #             yield ''.join(buffer)
    #             buffer.clear()
    #     if buffer:
    #         yield ''.join(buffer)

    return Response(response['output'])


if __name__ == '__main__':
    app.run(port=5000)


#response = agent.invoke({"input": "сколько бюджетных мест на программе “Информатика и вычислительная техника” "})