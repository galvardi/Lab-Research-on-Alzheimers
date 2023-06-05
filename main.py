import data_processor
import data_analyser
import pandas as pd

attributes_file_path = "Attributes_2_Read_CF.csv"
data_file_path = "small_data.csv"

if __name__ == '__main__':
    data = pd.read_csv(data_file_path).fillna("nan")

    analyser = data_analyser.DataAnalyser(data)
    processor = data_processor.DataProcessor(data, analyser)
    #
    processor.process_data()
    a = 5
    flag = False
    for col in processor.new_data:
        flag = pd.api.types.is_string_dtype(processor.new_data[col].dtype)
    print(flag)
