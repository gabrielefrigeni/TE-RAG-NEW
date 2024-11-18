import chromadb
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.core import Settings, VectorStoreIndex
import pandas as pd
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv
from models.gcp_client import init_gcp_client
import os
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

load_dotenv()
credentials = init_gcp_client()

class BuildNodes:
    """
    Class for building the Chroma vector database.
    """

    def __init__(self) -> None:
        self.df_assets = self.prepare_dataframe()
        self.collection_name = "demo"

    
    def prepare_dataframe(self) -> pd.DataFrame:
        """
        Pipeline for creating the final assets DataFrame

        Returns:
            pd.DataFrame: A merged DataFrame containing asset information
        """
        
        # Dati di esempio
        data = [
            {
                "nome asset": "anagrafica clienti",
                "tipo asset": "schema",
                "descrizione asset": "Schema che contiene le informazioni anagrafiche dei clienti della banca, inclusi dettagli personali e dati di contatto.",
                "schema di appartenenza": "",
                "descrizione schema": "",
                "tabella di appartenenza": "",
                "descrizione tabella di appartenenza": ""
            },
            {
                "nome asset": "clienti",
                "tipo asset": "tabella",
                "descrizione asset": "Tabella contenente i dati principali di identificazione dei clienti. Ogni riga rappresenta un cliente univoco con un identificativo.",
                "schema di appartenenza": "anagrafica clienti",
                "descrizione schema": "Schema che contiene le informazioni anagrafiche dei clienti della banca, inclusi dettagli personali e dati di contatto.",
                "tabella di appartenenza": "",
                "descrizione tabella di appartenenza": ""
            },
            {
                "nome asset": "codice_cliente",
                "tipo asset": "colonna",
                "descrizione asset": "Codice identificativo univoco assegnato a ciascun cliente della banca. Utilizzato per legare i record dei clienti con altre tabelle.",
                "schema di appartenenza": "anagrafica clienti",
                "descrizione schema": "Schema che contiene le informazioni anagrafiche dei clienti della banca, inclusi dettagli personali e dati di contatto.",
                "tabella di appartenenza": "clienti",
                "descrizione tabella di appartenenza": "Tabella contenente i dati principali di identificazione dei clienti. Ogni riga rappresenta un cliente univoco con un identificativo."
            },
            {
                "nome asset": "nome",
                "tipo asset": "colonna",
                "descrizione asset": "Nome del cliente. Campo testuale che contiene il nome proprio del cliente come indicato nei documenti di identificazione.",
                "schema di appartenenza": "anagrafica clienti",
                "descrizione schema": "Schema che contiene le informazioni anagrafiche dei clienti della banca, inclusi dettagli personali e dati di contatto.",
                "tabella di appartenenza": "clienti",
                "descrizione tabella di appartenenza": "Tabella contenente i dati principali di identificazione dei clienti. Ogni riga rappresenta un cliente univoco con un identificativo."
            },
            {
                "nome asset": "cognome",
                "tipo asset": "colonna",
                "descrizione asset": "Cognome del cliente. Campo testuale che contiene il cognome del cliente come indicato nei documenti di identificazione.",
                "schema di appartenenza": "anagrafica clienti",
                "descrizione schema": "Schema che contiene le informazioni anagrafiche dei clienti della banca, inclusi dettagli personali e dati di contatto.",
                "tabella di appartenenza": "clienti",
                "descrizione tabella di appartenenza": "Tabella contenente i dati principali di identificazione dei clienti. Ogni riga rappresenta un cliente univoco con un identificativo."
            },
            {
                "nome asset": "data_nascita",
                "tipo asset": "colonna",
                "descrizione asset": "Data di nascita del cliente, formattata come 'YYYY-MM-DD'. Essenziale per verificare l'età e l'idoneità a determinati servizi.",
                "schema di appartenenza": "anagrafica clienti",
                "descrizione schema": "Schema che contiene le informazioni anagrafiche dei clienti della banca, inclusi dettagli personali e dati di contatto.",
                "tabella di appartenenza": "clienti",
                "descrizione tabella di appartenenza": "Tabella contenente i dati principali di identificazione dei clienti. Ogni riga rappresenta un cliente univoco con un identificativo."
            },
            {
                "nome asset": "via",
                "tipo asset": "colonna",
                "descrizione asset": "Via di residenza del cliente. Campo testuale che indica il nome della strada come riportato nei documenti ufficiali.",
                "schema di appartenenza": "anagrafica clienti",
                "descrizione schema": "Schema che contiene le informazioni anagrafiche dei clienti della banca, inclusi dettagli personali e dati di contatto.",
                "tabella di appartenenza": "clienti",
                "descrizione tabella di appartenenza": "Tabella contenente i dati principali di identificazione dei clienti. Ogni riga rappresenta un cliente univoco con un identificativo."
            },
            {
                "nome asset": "numero_civico",
                "tipo asset": "colonna",
                "descrizione asset": "Numero civico associato all’indirizzo di residenza del cliente. Identifica il luogo preciso della residenza all’interno della via specificata.",
                "schema di appartenenza": "anagrafica clienti",
                "descrizione schema": "Schema che contiene le informazioni anagrafiche dei clienti della banca, inclusi dettagli personali e dati di contatto.",
                "tabella di appartenenza": "clienti",
                "descrizione tabella di appartenenza": "Tabella contenente i dati principali di identificazione dei clienti. Ogni riga rappresenta un cliente univoco con un identificativo."
            },
            {
                "nome asset": "città",
                "tipo asset": "colonna",
                "descrizione asset": "Città di residenza del cliente. Campo testuale che indica il comune di residenza del cliente.",
                "schema di appartenenza": "anagrafica clienti",
                "descrizione schema": "Schema che contiene le informazioni anagrafiche dei clienti della banca, inclusi dettagli personali e dati di contatto.",
                "tabella di appartenenza": "clienti",
                "descrizione tabella di appartenenza": "Tabella contenente i dati principali di identificazione dei clienti. Ogni riga rappresenta un cliente univoco con un identificativo."
            },
            {
                "nome asset": "CAP",
                "tipo asset": "colonna",
                "descrizione asset": "Codice di avviamento postale associato all’indirizzo del cliente, utilizzato per identificare la zona geografica di residenza.",
                "schema di appartenenza": "anagrafica clienti",
                "descrizione schema": "Schema che contiene le informazioni anagrafiche dei clienti della banca, inclusi dettagli personali e dati di contatto.",
                "tabella di appartenenza": "clienti",
                "descrizione tabella di appartenenza": "Tabella contenente i dati principali di identificazione dei clienti. Ogni riga rappresenta un cliente univoco con un identificativo."
            },
            {
                "nome asset": "numero_telefono",
                "tipo asset": "colonna",
                "descrizione asset": "Numero di telefono del cliente, utilizzato per contattare il cliente per questioni bancarie o di servizio. Formattato come 'XXX-XXX-XXXX'.",
                "schema di appartenenza": "anagrafica clienti",
                "descrizione schema": "Schema che contiene le informazioni anagrafiche dei clienti della banca, inclusi dettagli personali e dati di contatto.",
                "tabella di appartenenza": "clienti",
                "descrizione tabella di appartenenza": "Tabella contenente i dati principali di identificazione dei clienti. Ogni riga rappresenta un cliente univoco con un identificativo."
            },
            {
                "nome asset": "email",
                "tipo asset": "colonna",
                "descrizione asset": "Indirizzo email del cliente, usato per l'invio di comunicazioni e notifiche digitali della banca. Campo testuale.",
                "schema di appartenenza": "anagrafica clienti",
                "descrizione schema": "Schema che contiene le informazioni anagrafiche dei clienti della banca, inclusi dettagli personali e dati di contatto.",
                "tabella di appartenenza": "clienti",
                "descrizione tabella di appartenenza": "Tabella contenente i dati principali di identificazione dei clienti. Ogni riga rappresenta un cliente univoco con un identificativo."
            },
            {
                "nome asset": "anagrafica prodotti",
                "tipo asset": "schema",
                "descrizione asset": "Schema che contiene le informazioni sui prodotti bancari offerti, inclusi dettagli su conti, prestiti e carte di credito.",
                "schema di appartenenza": "",
                "descrizione schema": "",
                "tabella di appartenenza": "",
                "descrizione tabella di appartenenza": ""
            },
            {
                "nome asset": "prodotti",
                "tipo asset": "tabella",
                "descrizione asset": "Tabella contenente informazioni generali sui prodotti finanziari disponibili. Ogni riga rappresenta un prodotto univoco della banca.",
                "schema di appartenenza": "anagrafica prodotti",
                "descrizione schema": "Schema che contiene le informazioni sui prodotti bancari offerti, inclusi dettagli su conti, prestiti e carte di credito.",
                "tabella di appartenenza": "",
                "descrizione tabella di appartenenza": ""
            },
            {
                "nome asset": "codice_prodotto",
                "tipo asset": "colonna",
                "descrizione asset": "Codice univoco che identifica ogni prodotto bancario. Utilizzato per collegare i prodotti con altre tabelle o per reportistica.",
                "schema di appartenenza": "anagrafica prodotti",
                "descrizione schema": "Schema che contiene le informazioni sui prodotti bancari offerti, inclusi dettagli su conti, prestiti e carte di credito.",
                "tabella di appartenenza": "prodotti",
                "descrizione tabella di appartenenza": "Tabella contenente informazioni generali sui prodotti finanziari disponibili. Ogni riga rappresenta un prodotto univoco della banca."
            },
            {
                "nome asset": "nome_prodotto",
                "tipo asset": "colonna",
                "descrizione asset": "Nome del prodotto bancario, come 'Conto Corrente Base' o 'Prestito Personale'.",
                "schema di appartenenza": "anagrafica prodotti",
                "descrizione schema": "Schema che contiene le informazioni sui prodotti bancari offerti, inclusi dettagli su conti, prestiti e carte di credito.",
                "tabella di appartenenza": "prodotti",
                "descrizione tabella di appartenenza": "Tabella contenente informazioni generali sui prodotti finanziari disponibili. Ogni riga rappresenta un prodotto univoco della banca."
            },
            {
                "nome asset": "tipo_prodotto",
                "tipo asset": "colonna",
                "descrizione asset": "Tipologia del prodotto, ad esempio 'conto corrente', 'prestito' o 'carta di credito'.",
                "schema di appartenenza": "anagrafica prodotti",
                "descrizione schema": "Schema che contiene le informazioni sui prodotti bancari offerti, inclusi dettagli su conti, prestiti e carte di credito.",
                "tabella di appartenenza": "prodotti",
                "descrizione tabella di appartenenza": "Tabella contenente informazioni generali sui prodotti finanziari disponibili. Ogni riga rappresenta un prodotto univoco della banca."
            },
            {
                "nome asset": "tasso_interesse",
                "tipo asset": "colonna",
                "descrizione asset": "Tasso di interesse applicabile al prodotto bancario, espresso in percentuale. Rilevante per prestiti e conti con interessi.",
                "schema di appartenenza": "anagrafica prodotti",
                "descrizione schema": "Schema che contiene le informazioni sui prodotti bancari offerti, inclusi dettagli su conti, prestiti e carte di credito.",
                "tabella di appartenenza": "prodotti",
                "descrizione tabella di appartenenza": "Tabella contenente informazioni generali sui prodotti finanziari disponibili. Ogni riga rappresenta un prodotto univoco della banca."
            },
            {
                "nome asset": "commissione_annuale",
                "tipo asset": "colonna",
                "descrizione asset": "Commissione annuale associata al prodotto bancario, applicata generalmente a carte di credito e conti premium.",
                "schema di appartenenza": "anagrafica prodotti",
                "descrizione schema": "Schema che contiene le informazioni sui prodotti bancari offerti, inclusi dettagli su conti, prestiti e carte di credito.",
                "tabella di appartenenza": "prodotti",
                "descrizione tabella di appartenenza": "Tabella contenente informazioni generali sui prodotti finanziari disponibili. Ogni riga rappresenta un prodotto univoco della banca."
            },
            {
                "nome asset": "limite_credito",
                "tipo asset": "colonna",
                "descrizione asset": "Limite massimo di credito disponibile per il prodotto. Rilevante per carte di credito e linee di credito.",
                "schema di appartenenza": "anagrafica prodotti",
                "descrizione schema": "Schema che contiene le informazioni sui prodotti bancari offerti, inclusi dettagli su conti, prestiti e carte di credito.",
                "tabella di appartenenza": "prodotti",
                "descrizione tabella di appartenenza": "Tabella contenente informazioni generali sui prodotti finanziari disponibili. Ogni riga rappresenta un prodotto univoco della banca."
            },
            {
                "nome asset": "durata_minima",
                "tipo asset": "colonna",
                "descrizione asset": "Durata minima del contratto per il prodotto, espressa in mesi. Rilevante per prestiti e alcuni tipi di conti vincolati.",
                "schema di appartenenza": "anagrafica prodotti",
                "descrizione schema": "Schema che contiene le informazioni sui prodotti bancari offerti, inclusi dettagli su conti, prestiti e carte di credito.",
                "tabella di appartenenza": "prodotti",
                "descrizione tabella di appartenenza": "Tabella contenente informazioni generali sui prodotti finanziari disponibili. Ogni riga rappresenta un prodotto univoco della banca."
            },
            {
                "nome asset": "anagrafica filiali",
                "tipo asset": "schema",
                "descrizione asset": "Schema che raccoglie tutte le informazioni relative alle filiali della banca, incluse le informazioni di localizzazione, contatti e manager.",
                "schema di appartenenza": "",
                "descrizione schema": "",
                "tabella di appartenenza": "",
                "descrizione tabella di appartenenza": ""
            },
            {
                "nome asset": "filiali",
                "tipo asset": "tabella",
                "descrizione asset": "Tabella principale dello schema anagrafica filiali, contenente l'elenco di tutte le filiali con informazioni di base quali nome, codice, indirizzo e stato di apertura.",
                "schema di appartenenza": "anagrafica filiali",
                "descrizione schema": "Schema che raccoglie tutte le informazioni relative alle filiali della banca, incluse le informazioni di localizzazione, contatti e manager.",
                "tabella di appartenenza": "",
                "descrizione tabella di appartenenza": ""
            },
            {
                "nome asset": "codice_filiale",
                "tipo asset": "colonna",
                "descrizione asset": "Codice identificativo univoco della filiale, utilizzato internamente per identificare ciascuna sede.",
                "schema di appartenenza": "anagrafica filiali",
                "descrizione schema": "Schema che raccoglie tutte le informazioni relative alle filiali della banca, incluse le informazioni di localizzazione, contatti e manager.",
                "tabella di appartenenza": "filiali",
                "descrizione tabella di appartenenza": "Tabella principale dello schema anagrafica filiali, contenente l'elenco di tutte le filiali con informazioni di base quali nome, codice, indirizzo e stato di apertura."
            },
            {
                "nome asset": "nome_filiale",
                "tipo asset": "colonna",
                "descrizione asset": "Nome ufficiale della filiale, generalmente corrispondente alla città o alla zona di riferimento della sede.",
                "schema di appartenenza": "anagrafica filiali",
                "descrizione schema": "Schema che raccoglie tutte le informazioni relative alle filiali della banca, incluse le informazioni di localizzazione, contatti e manager.",
                "tabella di appartenenza": "filiali",
                "descrizione tabella di appartenenza": "Tabella principale dello schema anagrafica filiali, contenente l'elenco di tutte le filiali con informazioni di base quali nome, codice, indirizzo e stato di apertura."
            },
            {
                "nome asset": "indirizzo",
                "tipo asset": "colonna",
                "descrizione asset": "Indirizzo fisico della filiale, comprendente via, numero civico, città e CAP, utile per localizzare esattamente la posizione della sede.",
                "schema di appartenenza": "anagrafica filiali",
                "descrizione schema": "Schema che raccoglie tutte le informazioni relative alle filiali della banca, incluse le informazioni di localizzazione, contatti e manager.",
                "tabella di appartenenza": "filiali",
                "descrizione tabella di appartenenza": "Tabella principale dello schema anagrafica filiali, contenente l'elenco di tutte le filiali con informazioni di base quali nome, codice, indirizzo e stato di apertura."
            },
            {
                "nome asset": "manager",
                "tipo asset": "colonna",
                "descrizione asset": "Nome del manager responsabile della filiale, con il compito di supervisionare le operazioni e gestire il personale.",
                "schema di appartenenza": "anagrafica filiali",
                "descrizione schema": "Schema che raccoglie tutte le informazioni relative alle filiali della banca, incluse le informazioni di localizzazione, contatti e manager.",
                "tabella di appartenenza": "filiali",
                "descrizione tabella di appartenenza": "Tabella principale dello schema anagrafica filiali, contenente l'elenco di tutte le filiali con informazioni di base quali nome, codice, indirizzo e stato di apertura."
            }
        ]

        # Creazione del dataframe
        df = pd.DataFrame(data)
        df['descrizione asset'] = df.apply(self.concatenate_row, axis=1)

        return df


    def concatenate_row(self, row: pd.Series) -> pd.Series:
        """
        Concatenate attribute of each row into its description.
 
        Args:
            row (pd.Series): a row of the initial dataframe.
        """
        # Getting original description
        result = 'Descrizione: ' + row['descrizione asset']

        result += '\nNome asset: ' + row['nome asset']
        result += '\nTipo asset: ' + row['tipo asset']
        result += '\nNome tabella: ' + row['tabella di appartenenza']
        result += '\nDescrizione tabella: ' + row['descrizione tabella di appartenenza']
        result += '\nSchema tabella: ' + row['schema di appartenenza']
        result += '\nDescrizione schema tabella: ' + row['descrizione schema']

        return result

    def run_builder(self) -> None:
        """
        Build the Chroma vector store and index.
        """
        Settings.embed_model = VertexTextEmbedding(
            model_name=os.getenv("EMBEDDING_MODEL"),
            project=os.getenv('GCP_PROJECT_ID'),
            location=os.getenv('GCP_REGION'),
            credentials=credentials,
        )

        # Build a list of documents from the processed dataframe
        docs = list(
            map(
                lambda x: Document(
                    text=x[0],
                    metadata=x[1],
                    excluded_embed_metadata_keys=[x for x in self.df_assets.columns if x != 'Descrizione'],
                    excluded_llm_metadata_keys=[],
                    text_template="{metadata_str}\nDescrizione asset: {content}",
                    metadata_template='{key}: "{value}"',
                    metadata_seperator="\n",
                ),
                zip(
                    self.df_assets['descrizione asset'].values,
                    self.df_assets[[col for col in self.df_assets if col != 'descrizione asset']].to_dict(orient='records')
                )
            )
        )

        # Instantiate the Chroma client
        db = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH"))
        all_texts = [doc.text for doc in docs]

        # Embed the descriptions in batches
        batch_embeddings = Settings.embed_model.get_text_embedding_batch(texts=all_texts, show_progress=True)
        for i, doc in enumerate(docs):
            # Assign to each document its corresponding embedding
            doc.embedding = batch_embeddings[i]

        # Create a collection
        chroma_collection = db.get_or_create_collection(f'{self.collection_name}')
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=Settings.embed_model)

        for doc in tqdm(docs):
            # Insert each document in the index
            index.insert(doc)


def main():
    node_builder = BuildNodes()
    node_builder.run_builder()

if __name__ == '__main__':
    main()