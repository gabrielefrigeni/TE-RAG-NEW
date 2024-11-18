from google.oauth2 import service_account
from google.auth.transport.requests import Request
import vertexai
import os

def init_gcp_client():
    """
    Initialize the GCP client and connect to Vertex using the GCP credentials.

    Returns:
        credentials: the credentials to run Gemini using LlamaIndex.
    """
    filename = os.getenv('GCP_KEY_PATH')
    credentials: service_account.Credentials = (
        service_account.Credentials.from_service_account_file(filename)
    )

    if credentials.expired:
        credentials.refresh(Request())

    vertexai.init(project=os.getenv('PROJECT_ID'), location=os.getenv('REGION'), credentials=credentials)
    return credentials