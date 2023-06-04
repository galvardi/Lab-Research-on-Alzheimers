import data_processor
import data_analyser
import pandas as pd

csv_file_path = "C:/Users/tomer/OneDrive/Documents/Academy/year 3/semesterB/LAB/Filemail.com files 6_2_2023/temp.csv"
if __name__ == '__main__':
    data = pd.read_csv(csv_file_path).fillna('Prefer not to answer')

    analyser = data_analyser.DataAnalyser(data)
    processor = data_processor.DataProcessor(data, analyser)

    processor.process_data()
    a = 5
