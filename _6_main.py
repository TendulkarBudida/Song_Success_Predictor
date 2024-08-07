import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('C:\\Song_Success_Predictor\\trained_model.keras')

# Load the encoder
encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def predict_rating(lyrics):
    # Encode the lyrics
    encoded_lyrics = encoder.encode(lyrics, normalize_embeddings=True)
    
    # Reshape the input to match the expected shape of the model
    X = np.array([encoded_lyrics])  # Wrap the encoded lyrics in a list to create a batch of size 1
    
    # Predict the rating
    predicted_rating = model.predict(X)
    
    # Convert the predicted rating to a readable format (assuming it's a single label)
    predicted_rating_label = np.argmax(predicted_rating)  # Assuming softmax output
    return predicted_rating_label


# Example usage
input_lyrics = """Somebody told me it was pointless for me to come back
Into your arms
Said you fucked another man, finally
I knew this day would come, woah oh, oh
'Cause I see fear in your eyes
You've been living out your life
As long as you know that when I land you're mine (mhm, mhm)
It's been exactly three sixty-five since I've seen your face (あったかいね)
I've been living on the road
And you've been living all alone, at home
Girl I hope
He made you satisfied (気持ちいい)
Well, baby I won't cry
As long as you know that when I land you're mine
And you will never feel so pretty
And you will never feel this beautiful
When I make it there
Oh, when I make it there
There are certain things that I've come to understand (ah, ooh)
Expectations can kill a simple man, simple woman woah oh
I try to master the art
Of that far away love
But only so much can keep a woman warm
(Warm, ooh)
Now it's times like this that I say to myself
(Say to myself)
We've been living in a cold, cold world
Cold world
But at least I have you to rely (気持ちいい)
Even if for a short time (うん 気持ちいい)
As long as you know when I land you're mine
And you will never feel so pretty
And you will never feel this beautiful
Oh, when I make it there
Oh, when I make it there
And he can't make you feel this pretty
No, he won't make you feel this beautiful
When I make it there
Oh, when I make it there
Ah (no, no-no)
Ah (no, no-no)
Ah (no, no, no, no, no)
Ah (oh oh oh ah)
Ooh (hoo, hoo, hoo)
Ooh (hoo, hoo, hoo)
Ooh (oh babe) (oh, oh)
Ooh, oh
Ooh, oh
Ooh, oh, oh
And you will never feel so pretty (pretty)
And you will never feel this beautiful (beautiful, oh)
When I make it there (when I make it there)
Oh when I make it there (hey)
And he can't make you feel this pretty (pretty)
No he won't make you feel this beautiful (beautiful) (hoo, hoo, hoo)
When I make it there (when I make it there baby) (oh baby)
Oh, when I make it there (hoo, hoo, hoo)
And you will never feel so pretty (pretty) (oh babe, no you won't)
And you will never feel this beautiful (beautiful) (hoo, hoo, hoo, eh)
When I make it there (when I make it there) (hey)
Oh, when I make it there
And he can't make you feel this pretty (baby)
And he won't make you feel this beautiful (oh babe)
When I make it there (when I make it there) (hey)
When I make it there
When I make it there"""

predicted_rating = predict_rating(input_lyrics)
print("Predicted Rating:", predicted_rating)