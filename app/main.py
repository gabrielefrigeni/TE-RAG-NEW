import chainlit as cl
from utils.user_output import format_source
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from chat_engine.load_chat_engine import load_chat_engine
from common.prompts_templates.PromptTemplates import (
    HELLO_MESSAGE,
    EMPTY_SOURCES_MESSAGE
)
import os

####################### A small note #############################################
# Some lines of code are commented out in this file.
# This is because they call the asset mapping function, which is currently not supported given our knowledge base structure.
# Just ignore them
##################################################################################

# Function triggered when the Chainlit application is launched
@cl.on_chat_start
async def start_chat():
    response_msg = cl.Message(content=HELLO_MESSAGE, author=os.getenv("AUTHOR"))
    await response_msg.send()
    
    # Load the chat engine, the core LlamaIndex component
    cl.user_session.set("query_engine", load_chat_engine())
    cl.user_session.set("memory", [])
    cl.user_session.set("assets", [])


# Function triggered every time the chat receives a new input message from the user
@cl.on_message
async def handle_msg(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")
    memory = cl.user_session.get("memory")

    response_msg = cl.Message(content="", author=os.getenv("AUTHOR"))
    await response_msg.send()

    response = await cl.make_async(query_engine.stream_chat)(message=message.content, chat_history=memory)
    response_msg.content = response.response

    for token in response.response_gen:
        await response_msg.stream_token(token)
    await response_msg.send()

    sources = [n for n in response.source_nodes if n.score]
    if sources:
        source_refs = r"\, ".join([f"Fonte {idx+1}" for idx, _ in enumerate(sources)])
        source_elem = [cl.Text(name=f"Fonte {idx+1}",
                            content=f"{format_source(source) or 'Empty node'}",
                            display="side") for idx, source in enumerate(sources)]
        response_msg.content += f"\n\n Fonti: {source_refs}"
        response_msg.elements = source_elem

    memory.append(ChatMessage(role=MessageRole.USER, content=message.content))

    # If the reranker returned no documents, the response is empty
    if response.response == 'Empty Response':
        response_msg.content = EMPTY_SOURCES_MESSAGE
    else:
        memory.append(ChatMessage(role=MessageRole.ASSISTANT, content=response_msg.content))

    await response_msg.update()
    cl.user_session.set("memory", memory)


# the following code allows to run in debug mode on VSCode
if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)