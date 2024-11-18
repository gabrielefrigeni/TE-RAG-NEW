def format_source(source_node):
    """
    Format the retrieved sources to be displayed
    """
    SHOW_METADATA = ['Nome asset', 'Tipo asset', 'Descrizione', 'Nome tabella', 'Schema']
    output_str = ""
    for k,v in source_node.node.metadata.items():
        if k in SHOW_METADATA:
            output_str += f"**{(' '.join(k.split('_'))).capitalize()}**: {v}\n"
    output_str += f'\n\n**Descrizione asset:**\n{str(source_node.node.text)}'
    output_str += f'\n\n**Similarity score:**\n{str(source_node.score)}'
    return output_str





