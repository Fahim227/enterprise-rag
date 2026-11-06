import json
from functools import lru_cache


from app.aws.client import aws_client
from app.aws.models import EmbedRequest
from app.config import settings


class BedrockEmbedder:
    """Turns text into vectors with a Bedrock Titan embedding model.

    The boto3 client is passed in, so tests can supply a fake one.
    """

    def __init__(self, client, model_id: str, dimensions: int):
        self._client = client
        self._model_id = model_id
        self._dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError('Cannot embed empty text.')
        request = EmbedRequest(input_text=text, dimensions=self._dimensions)
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(request.to_map()),
            contentType='application/json',
            accept='application/json',
        )
        vector = json.loads(response['body'].read())['embedding']
        if len(vector) != self._dimensions:
            raise RuntimeError(f'Expected {self._dimensions} dimensions, got {len(vector)}.')
        return vector


@lru_cache
def default_embedder() -> BedrockEmbedder:
    s = settings()
    return BedrockEmbedder(aws_client('bedrock-runtime'), s.bedrock_embed_model, s.embed_dim)


def embed(text: str) -> list[float]:
    return default_embedder().embed(text)
