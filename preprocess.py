from ast import literal_eval
from csv import reader
from os import listdir, makedirs, path
from pickle import dump
import numpy as np
from sklearn.model_selection import train_test_split
from args import get_parser
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_save(category, filename, dataset, dataset_folder, output_folder):

    if dataset == "df_abnormal_transac_0918":
        label_encoder = LabelEncoder()
        df['tran_code'] = label_encoder.fit_transform(df['tran_code'])
        print(df.info())
    elif dataset == "DF_ABNORMAL_METRIC_0918" :
        df = pd.read_csv(path.join(dataset_folder, filename))

        df =df.drop(df.columns[0], axis=1)
        label_encoder = LabelEncoder()
        #perform label encoding across team, position, and all_star columns
        df[['cmdb_id','kpi_name','device']] = df[['cmdb_id','kpi_name','device']].apply(LabelEncoder().fit_transform)
        print(df.info())
        df=df.astype(float)
        # Split the DataFrame into train and test sets
        train_df, test_df = train_test_split(df, test_size=0.2)
        # Save train set
        train_output_path = path.join(output_folder, f"{dataset}_train.pkl")
        with open(train_output_path, "wb") as train_file:
            dump(train_df.values, train_file)
        print(f"Saved train set: {train_output_path}")

        # Save test set
        test_output_path = path.join(output_folder, f"{dataset}_test.pkl")
        with open(test_output_path, "wb") as test_file:
            dump(test_df.values, test_file)
        print(f"Saved test set: {test_output_path}")
    else :
        temp = np.genfromtxt(
            path.join(dataset_folder, category, filename),
            dtype=np.float32,
            delimiter=",",
        )
        print(dataset, category, filename, temp.shape)
        with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
            dump(temp, file)


def load_data(dataset):
    """ Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly) """

    if dataset == "SMD":
        dataset_folder = "datasets/ServerMachineDataset"
        output_folder = "datasets/ServerMachineDataset/processed"
        makedirs(output_folder, exist_ok=True)
        file_list = listdir(path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith(".txt"):
                load_and_save(
                    "train",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )

                load_and_save(
                    "test_label",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )

                load_and_save(
                    "test",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )
                test_label = None

    elif dataset == "SMAP" or dataset == "MSL":
        dataset_folder = "datasets/data"
        output_folder = "datasets/data/processed"
        makedirs(output_folder, exist_ok=True)
        with open(path.join(dataset_folder, "labeled_anomalies.csv"), "r") as file:
            csv_reader = reader(file, delimiter=",")
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        data_info = [row for row in res if row[1] == dataset and row[0] != "P-2"]
        labels = []
        for row in data_info:
            anomalies = literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool_)
            for anomaly in anomalies:
                label[anomaly[0] : anomaly[1] + 1] = True
            labels.extend(label)

        labels = np.asarray(labels)
        print(dataset, "test_label", labels.shape)

        with open(path.join(output_folder, dataset + "_" + "test_label" + ".pkl"), "wb") as file:
            dump(labels, file)

        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(path.join(dataset_folder, category, filename + ".npy"))
                data.extend(temp)
            data = np.asarray(data)
            print(dataset, category, data.shape)
            with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
                dump(data, file)

        for c in ["train", "test"]:
            concatenate_and_save(c)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    ds = args.dataset.upper()
    load_data(ds)
