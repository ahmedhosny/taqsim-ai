"""Audio classification using the MAEST model."""

import numpy as np
from transformers import pipeline

# audio @16kHz
audio = np.random.randn(30 * 16000)

pipe = pipeline("audio-classification", model="mtg-upf/discogs-maest-30s-pw-129e", trust_remote_code=True)
result = pipe(audio)

# Print the classification results
print("\nAudio Classification Results:")
for prediction in result[:10]:
    print(f"Label: {prediction['label']}, Score: {prediction['score']:.4f}")


