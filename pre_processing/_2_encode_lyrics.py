import pandas as pd # encoding and creating a new csv file
from sentence_transformers import SentenceTransformer

class LyricsEncoder:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.embedding = None

    def encode(self, lyrics):
        """Transform lyrics into vectors"""
        embedding = self.model.encode(lyrics, normalize_embeddings=True)
        self.embedding = embedding
        return embedding

encoder = LyricsEncoder()

# Function to encode lyrics and add them to a new column
def encode_lyrics(row):
    encoded_lyrics = encoder.encode(row['lyrics'])
    return encoded_lyrics

# Read the CSV file
df = pd.read_csv('C:\\Song_Success_Predictor\\pre_processing\\selected_columns.csv')

# Apply the function to each row and create a new column 'encoded_lyrics'
df['encoded_lyrics'] = df.apply(encode_lyrics, axis=1)
print(df['encoded_lyrics'].dtype)
# Write the DataFrame back to the CSV file
df.to_csv('C:\\Song_Success_Predictor\\pre_processing\\selected_columns.csv', index=False)