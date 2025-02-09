import pandas as pd

class IngestData:
    def __init__(self) -> None:
        """
        Initializes the IngestData class with an empty data path.
        """
        self.data_path = None

    def get_data(self, data_path: str) -> pd.DataFrame:
        """
        This function takes a path to a CSV file and returns a pandas DataFrame.
        Parameters:
        - data_path (str): The path to the CSV file to be ingested.
        Returns:
        - pd.DataFrame: The loaded data in the form of a pandas DataFrame.
        """
        self.data_path = data_path
        df = pd.read_csv(self.data_path)
        return df
