from collections import Counter
import os
import csv


class DesignationMatching:
    def __init__(self, data):
        self.data = data

    def actual_des(self):
        designation_counts = Counter(self.data['Designation'])
        print(designation_counts)

    def assign_priority(self):
        counts = [0] * len(self.data)
        for designation in self.data.Designation.unique():
            count = 0
            for idx in range(len(self.data)):
                if self.data.iloc[idx]['Designation'] == designation:
                    count += 1
                    counts[idx] = count
        self.data['Priority'] = counts

    def designation_matching(self):
        result_str = ''
        for priority in self.data.Priority.unique():
            sub_data = self.data[self.data.Priority == priority]
            result_str += f'Priority--{priority}\n'
            for idx in range(len(sub_data)):
                result_str += f"{sub_data.iloc[idx]['Ename']} {sub_data.iloc[idx]['Designation']}\n"
            result_str += '--------------------------------------------------------------\n'
        return result_str


def write_to_csv(file_path, fields, header=None):
    mode = 'a' if os.path.exists(file_path) else 'w'

    with open(file_path, mode, newline='') as f:
        writer = csv.writer(f)
        if mode == 'w' and header:
            writer.writerow(header)
        writer.writ
