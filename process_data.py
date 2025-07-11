import csv
import joblib
import numpy as np
from sklearn.model_selection import train_test_split


def returnLLM(llm_name):
    if llm_name == "meta-llama/Llama-2-70b-hf":
        return 0.0
    elif llm_name == "meta-llama/Llama-2-7b-hf":
        return 1.0
    else:
        return 2.0


def returnGPU(gpu_name):
    if gpu_name == "a100":
        return 0.0
    else:
        return 1.0


def makeEntry(row):
    llm = returnLLM(row[0].strip())
    first_data = np.expand_dims(np.array(llm, dtype=np.float32), axis=0).reshape(1, 1)
    gpu = returnGPU(row[1].strip())
    second_data = np.expand_dims(np.array(gpu, dtype=np.float32), axis=0).reshape(1, 1)
    combined = np.concatenate((first_data, second_data), axis=1)
    for i in range(2, 6):
        temp = float(row[i].strip())
        second_data = np.expand_dims(np.array(temp, dtype=np.float32), axis=0).reshape(
            1, 1
        )
        combined = np.concatenate((combined, second_data), axis=1)

    temp = float(row[6].strip())
    label_data_0 = np.expand_dims(
        np.array(temp / 1.92, dtype=np.float32), axis=0
    ).reshape(1, 1)

    temp = float(row[7].strip())
    label_data_1 = np.expand_dims(
        np.array(temp / 40.3, dtype=np.float32), axis=0
    ).reshape(1, 1)
    combined_label = np.concatenate((label_data_0, label_data_1), axis=1)

    return combined, combined_label


if __name__ == "__main__":
    cnt = [0, 0, 0]
    counter = [3334, 3333, 3333]
    with open("./data0/data.csv", "r", newline="\n") as file:
        reader = csv.reader(file)
        first_entry = None
        first_label = None
        for row in reader:
            second_entry, second_label = makeEntry(row)
            index = int(second_entry[0][0])
            if cnt[index] >= counter[index]:
                continue
            else:
                cnt[index] += 1

            if first_entry is None:
                first_entry = second_entry
            else:
                first_entry = np.concatenate((first_entry, second_entry), axis=0)

            if first_label is None:
                first_label = second_label
            else:
                first_label = np.concatenate((first_label, second_label), axis=0)

        # if first_entry is not None:
        #     print(first_entry.shape)
        #     print(first_entry.dtype)

        # if first_label is not None:
        #     print(first_label.shape)
        #     print(first_label.dtype)

        file.close()

        if first_entry is not None and first_label is not None:
            X = first_entry
            y = first_label

            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )

            X_test, X_val, y_test, y_val = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=88, shuffle=True
            )

            print(X_train.shape)
            print(X_train.shape)
            print(X_test.shape)
            print(X_val.shape)
            print(y_train.shape)
            print(y_test.shape)
            print(y_val.shape)

            cnt = [0, 0, 0]
            test_entry_buck = [[], [], []]
            test_label_buck = [[], [], []]
            for i in range(X_test.shape[0]):
                index = int(X_test[i][0])
                cnt[index] += 1
                test_entry_buck[index].append(X_test[i])
                test_label_buck[index].append(y_test[i])
            print(cnt)

            test_data = []
            test_label = []
            for i in range(3):
                test_data.append(np.stack(test_entry_buck[i], axis=0))
                test_label.append(np.stack(test_label_buck[i], axis=0))
            
            for i in range(3):
                print(test_data[i].shape)
                print(test_label[i].shape)

            joblib.dump(
                [X_train, X_val, test_data, y_train, y_val, test_label], "fake_data_all.joblib"
            )
