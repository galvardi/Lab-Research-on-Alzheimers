import numpy as np

import data_processor
import data_analyser
import pandas as pd

attributes_file_path = "Attributes_2_Read_CF.csv"
data_file_path = "saves/small_data.csv"

if __name__ == '__main__':
    data = pd.read_csv(data_file_path).fillna("nan")

    analyser = data_analyser.DataAnalyser(data)
    processor = data_processor.DataProcessor(data, analyser)
    #
    processor.process_data()
    # print(processor.new_data.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()))
    # get only the rows with column "years_from_rec_to_diagnosis" > 7 and < 14
    print("{")
    for i, col in enumerate(processor.new_data.columns):
        print(str(i) + ": '" + col + "',")
    print("}")
    processor.new_data.to_csv("saves/new_data.csv", index=False)
    a = 3
