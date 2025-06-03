import csv
import pandas as pd

ori_train_path = "Dataset_4_ori\\sign_data\\train_data.csv"
ori_test_path = "Dataset_4_ori\\sign_data\\test_data.csv"
full_set_path = "Dataset_4_ori\\sign_data\\full_dataset.csv"

train_data = pd.read_csv(ori_train_path)
print(train_data)
test_data = pd.read_csv(ori_test_path)
# print(test_data.size)

full_data = pd.concat([train_data, test_data], ignore_index=True).drop_duplicates()

with open(full_set_path, mode='w+', newline="") as file:
    writer = csv.writer(file)
    writer.writerows(full_data.values.tolist())
