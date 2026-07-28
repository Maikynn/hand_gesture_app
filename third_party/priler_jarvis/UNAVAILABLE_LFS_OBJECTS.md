# Unavailable upstream LFS objects

The following Priler/Jarvis objects could not be mirrored because the
upstream repository's public Git LFS budget was exhausted. GitHub rejects
unknown LFS pointers in a new repository, so the pointer files themselves are
not included in this snapshot.

| Upstream path | SHA-256 | Size |
| --- | --- | ---: |
| `resources/models/all-MiniLM-L6-v2/model.onnx` | `bbd7b466f6d58e646fdc2bd5fd67b2f5e93c0b687011bd4548c420f7bd46f0c5` | 90,387,630 bytes |
| `resources/models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/model.onnx` | `634d0f66c29dc934c8fa72b8a4fe91dd4d420a22f1d82a241058d4316e659a99` | 235,052,644 bytes |
| `resources/models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/tokenizer.json` | `fa685fc160bbdbab64058d4fc91b60e62d207e8dc60b9af5c002c5ab946ded00` | 17,083,009 bytes |
| `resources/models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/unigram.json` | `da145b5e7700ae40f16691ec32a0b1fdc1ee3298db22a31ea55f57a966c4a65d` | 14,763,260 bytes |

These sentence-transformer assets are not used by the native Python
integration in Axi Control. The application uses editable fuzzy matching for
commands and the complete Russian Vosk model for offline speech recognition.
