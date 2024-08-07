# Song Success Predictor

This project aims to predict the success rating of songs based on their lyrics using machine learning techniques. The project involves data preprocessing, model training, and prediction.

## Project Structure

## Files and Scripts
- **_5_model_training.py**: Script to train the machine learning model.
- **_6_main.py**: Script to predict the rating of a song based on its lyrics.
- **jup.ipynb**: Jupyter notebook for exploratory data analysis and experimentation.
- **pre_processing/**: Directory containing scripts for data preprocessing.
  - **_1_get_columns.py**: Extracts specific columns from the raw data.
  - **_2_encode_lyrics.py**: Encodes song lyrics using a pre-trained model.
  - **_3_lyrics_to_float.py**: Converts encoded lyrics to float values.
  - **_4_get_rating.py**: Retrieves the rating for each song.
  - **selected_columns.csv**: Preprocessed CSV file with selected columns.
  - **song_lyrics_sample.csv**: Sample CSV file with raw song lyrics data.
- **trained_model.keras**: The trained machine learning model.

## Setup and Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd Song_Success_Predictor
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Data Preprocessing

1. Extract specific columns from the raw data:
    ```sh
    python pre_processing/_1_get_columns.py
    ```

2. Encode the lyrics:
    ```sh
    python pre_processing/_2_encode_lyrics.py
    ```

## Model Training

Train the model using the preprocessed data:
```sh
python _5_model_training.py