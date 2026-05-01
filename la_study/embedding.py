from google import genai
from google.genai import types
import os

os.environ['GEMINI_API_KEY'] = INSERT YOUR OWN

temodel = "text-embedding-3-large"
# see platform.openai.com/docs/guides/embeddings

def adaEmbedding(a):
    string = json.loads(a)
    response = client.embeddings.create(input=string, model=temodel)
    embedding = response.data[0].embedding
    return(json.dumps(embedding))

server.register_function(adaEmbedding)