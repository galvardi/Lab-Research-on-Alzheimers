import data_processor
import data_analyser
import pandas as pd

attributes_file_path = "Attributes_2_Read_CF.csv"
data_file_path = "small_data.csv"

if __name__ == '__main__':
    data = pd.read_csv(data_file_path).fillna('Prefer not to answer')

    analyser = data_analyser.DataAnalyser(data)
    # processor = data_processor.DataProcessor(data, analyser)
    #
    # processor.process_data()
    # a = 5
