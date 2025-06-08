import os
import pandas as pd

def process_labels(lines):
    """
    Processes the labels from a list of lines.
    Args:
        lines (list): A list of strings, each representing a line from a file.
    Returns:
        list: A list of lists, where each inner list contains integers parsed from the line.
    """
    return [[int(x) for x in line.strip().split()] for line in lines]

def process_utterances(lines):
    """
    Processes the utterances from a list of lines.
    Args:
        lines (list): A list of strings, each representing a line from a file.
    Returns:
        list: A list of lists, where each inner list contains utterances split by "__eou__".
    """
    return [
        [utt.strip() for utt in line.strip().split("__eou__") if utt.strip()]
        for line in lines
    ]

def create_csv(path: str) -> None:
    """
    Reads three txt files at a directory and creates a CSV file with the data.
    And saves it in the same directory.
    Args:
        path (str): The directory path where the txt files are located.
    Returns:
        None
    """

    # Check if the directory exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"The directory {path} does not exist.")
    
    # List of txt filesname and directory to read
    paths = ["train", "test", "validation"]
    txt_files = ['dialogues_act', 'dialogues_emotion', 'dialogues']

    for name in paths:
        # Create a list to hold the data
        data = []

        # Read each txt file and append its content to the data list
        for file in txt_files:
            file_path = os.path.join(path, f"{name}/{file}_{name}.txt")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file {file_path} does not exist.")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip().split('\n')
                data.append(content)
        '''
        acts = process_labels(data[0])
        emotions = process_labels(data[1])
        utterances = process_utterances(data[2])
        '''  

        # Create a DataFrame from the data
        df = pd.DataFrame({
            'acts': data[0],
            'emotions': data[1],
            'utterances': data[2]
        })

        # Save the DataFrame to a CSV file
        csv_file_path = os.path.join(path, f"{name}.csv")
        df.to_csv(csv_file_path, index=False)
        print(f"CSV file created: {csv_file_path}")
    return 


def main():
    # Define the path to the directory containing the txt files
    path = 'data/'

    # Create the CSV file from the txt files in the specified directory
    # create_csv(path)

    # test reading the CSV file
    df = pd.read_csv(os.path.join(path, 'train.csv'))
    # print(df.head(5))
    # check the columns element type (acts, emotions, utterances)
    print(df['acts'][0].split())

main()