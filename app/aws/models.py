from dataclasses import dataclass


@dataclass(frozen=True)
class EmbedRequest:
    input_text: str
    dimensions: int
    normalize: bool = True

    def to_map(self):
        # Titan Embeddings expects camelCase keys in the request body.
        return {
            'inputText': self.input_text,
            'dimensions': self.dimensions,
            'normalize': self.normalize
        }
