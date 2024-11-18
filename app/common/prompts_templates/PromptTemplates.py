SYSTEM_PROMPT_ASSISTANT="""
Always reply in italian.
Consider that you can offer information about:
-Description and usage of columns, tables and schemas contained inside the new ODS Data Warehouse;
-Search for columns and tables by their functional description;
-Where old columns and tables from the old SAS Data Warehouse have been mapped into the new ODS Data Warehouse and vice versa;
-Report problems within the data warehouse.
Always be kind and finish your answer by asking the user if you can help with something in particular, and refet to them using second person. 
Make a bullet list to show what they can ask.
""".strip()


SINGLE_SELECT_PROMPT = """
Some choices are given below. It is provided in a numbered list (1 to {num_choices}), where each item in the list corresponds to a summary.\n
You are allowed to select only 1 of the following tools. It is absolutely forbidden to output more than one selection.
\n---------------------\n
{context_list}
\n---------------------\n
Using only the choices above and not prior knowledge, return 1 and ONLY 1 choice that is most relevant to the follwing question:\n'{query_str}'\n
""".strip()


CONDENSE_QUESTION_PROMPT = """
Given a conversation (between Human and Assistant) and a follow up message from Human, rewrite the message including a brief summary of the context of the conversation. If you find asset names, include them in the summary. Be very specific about the followup question, the assistant's answer will need to be based on this.
<Chat History>
{chat_history}
<Follow Up Message>
{question}
<Standalone question>
""".strip()


TOOL_DESCRIPTIONS = {
    "semantic_search" : """Use this query engine if you need to lookup specific information about the data assets contained in the data warehouse {key_name}. Never use this query engine if you are asked to actually do something other than answering a question. Consider that "ODS" is the default data warehouse; consider "OCS" only if the question explicitly mentions it.""".strip(),
    "asset_mapping" : "Use this query engine if and only if the question explicitly asks about the mapping for an asset between two data warehouses. Never use this query engine if the question does not mention the SAS data warehouse and or a set of given assets.".strip(),
    "general_interaction" : "Use this query engine if the question is about something that does not involve data assets stored in a data warehouse or the functioning of the chatbot. Never use this query engine if the question is about data or metadata stored in a data warehouse or about what the chatbot can do.".strip(),
    "problems_reporting" : "Use this query engine when the question is reporting either specific or general issues about the data warehouse and its contents.".strip(),
    "chatbot_info" : "Use this query engine when the question is about the capabilities and functionalities of the chatbot, like 'hi, what can you do?', or 'what can I ask you?'.".strip()
}


HELLO_MESSAGE = """
Ciao, come posso aiutarti?
""".strip()


MAPPING_SEARCH_RESPONSE = "Ricerca mapping SAS-Oracle in corso..."


EMPTY_SOURCES_MESSAGE = """
Mi dispiace ma non ho trovato informazioni riguardo la tua domanda.
Prova a riformularla o a chiedermi qualcos'altro.
Se ritieni che la mancanza di queste informazioni sia un errore, vuoi che apra una richiesta di supporto a DMO?
""".strip()
