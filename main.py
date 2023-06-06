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
    # get only the rows with column "years_from_rec_to_diagnosis" > 7 and < 14
    a = 3
