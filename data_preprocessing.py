import re
import os
import numpy as np

SEP = '<eod>'
def read_trainingdata(file_path, target1, target2, target3, target4, target5):
    keys = []
    values = []
    dialog = []
    dialog_num = 0
    pattern1 = ' __eou__ '
    pattern2 = ' __eou__'
    with open(file_path, 'r') as f:
        for line in f:
            dialog_num += 1
            keys.append(dialog_num)
            speaker = re.split(pattern1, line)
            if speaker[-1] == '\n':
                speaker.remove('\n')
            sub = re.sub(pattern2, '', speaker[-1])
            del speaker[-1]
            speaker.append(sub)
            dialog.append(speaker)
            values.append(dialog)
            dialog = []
            speaker = []
    training_data = dict(zip(keys,values))
    with open (target1, 'w') as t, open (target3, 'w') as s:
        for values in training_data.values():
            for index in range(np.array(values).shape[1]):
                    if index != np.array(values).shape[1] - 1:
                        t.write(values[0][index] + '\n')
                        s.write('s' + '\n')
                    else:
                        t.write(values[0][index] + SEP + '\n')
                        s.write(SEP + '\n')

    with open (target2, 'w') as t:
        for values in training_data.values():
            for index in range(np.array(values).shape[1]):
                    if index != np.array(values).shape[1] - 1:
                        t.write(values[0][index] + '\n')
                    else:
                        t.write(values[0][index])

    with open (target4, 'w') as t, open (target5, 'w') as s:
        for values in training_data.values():
            if np.array(values).shape[1] % 2 == 0:
                for index in range(np.array(values).shape[1]):
                    if index % 2 == 0:
                        if index != np.array(values).shape[1] - 1:
                            t.write(values[0][index] + '\n')
                        else:
                            t.write(values[0][index])
                    else:
                        if index != np.array(values).shape[1] - 1:
                            s.write(values[0][index] + '\n')
                        else:
                            s.write(values[0][index])
            else:
                for index in range(np.array(values).shape[1] - 1):
                    if index % 2 == 0:
                        if index != np.array(values).shape[1] - 1:
                            t.write(values[0][index] + '\n')
                        else:
                            t.write(values[0][index])
                    else:
                        if index != np.array(values).shape[1] - 1:
                            s.write(values[0][index] + '\n')
                        else:
                            s.write(values[0][index])


def main():
    read_trainingdata('valid/dialogues_validation.txt', 'valid/valid_data.txt',
                    'valid/valid_data1.txt', 'valid/valid_data2.txt',
                    'valid/speaker.txt', 'valid/answer.txt')

if __name__ == "__main__":
    main()