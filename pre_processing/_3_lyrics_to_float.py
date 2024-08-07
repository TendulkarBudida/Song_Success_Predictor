import pandas as pd # converting into float values

# Load the DataFrame from CSV
df = pd.read_csv('C:\\Song_Success_Predictor\\pre_processing\\selected_columns.csv')

# Function to convert a string of encoded lyrics to a list of floats
def convert_encoded_lyrics(encoded_lyric):
    encoded_lyric = encoded_lyric[1:-1]
    encoded_lyric = encoded_lyric.split()
    encoded_lyric = [float(x) for x in encoded_lyric]
    return encoded_lyric

# Apply the conversion function to each row in the 'encoded_lyrics' column
df['encoded_lyrics'] = df['encoded_lyrics'].apply(convert_encoded_lyrics)

# Write the DataFrame back to the CSV file
df.to_csv('C:\\Song_Success_Predictor\\pre_processing\\selected_columns.csv', index=False)