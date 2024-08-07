import csv # for getting lyrics and views into a new csv file

def extract_columns(input_file, output_file, column_names, encoding='utf-8'):
    with open(input_file, 'r', newline='', encoding=encoding) as csv_input:
        reader = csv.DictReader(csv_input)
        
        # Extract only specified columns
        extracted_data = [{col: row[col] for col in column_names} for row in reader]
        
        # Write extracted data to a new CSV file
        with open(output_file, 'w', newline='', encoding=encoding) as csv_output:
            writer = csv.DictWriter(csv_output, fieldnames=column_names)
            writer.writeheader()
            writer.writerows(extracted_data)

# Example usage:
input_file = 'C:\\Song_Success_Predictor\\song_lyrics_sample.csv'
output_file = 'C:\\Song_Success_Predictor\\pre_processing\\selected_columns.csv'
columns_to_extract = ['lyrics', 'views']

extract_columns(input_file, output_file, columns_to_extract)