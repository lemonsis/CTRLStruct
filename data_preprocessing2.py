import re

SEP = '<eod>'

def readdata(filepath, target1, target2, target3, target4, target5):
    sen = []
    with open (filepath, 'r') as f, open (target1, 'w') as t, open (target3, 'w') as s:
        for line in f:
            if line[0] == str(1):
                new_line = ""
                for i in range (2, len(line)):
                    new_line += line[i]
                t.write(SEP)
                s.write(SEP)
                t.write('\n')
                s.write('\n')
                sen = new_line.split('\t')
                t.write(sen[0])
                s.write('s')
                t.write('\n')
                s.write('\n')
                t.write(sen[1])
                s.write('s')
                t.write('\n') 
                s.write('\n')
            else:
                new_line = ""
                for i in range (2, len(line)):
                    new_line += line[i]
                sen = new_line.split('\t')
                t.write(sen[0])
                s.write('s')
                t.write('\n')
                s.write('\n')
                t.write(sen[1])
                s.write('s')
                t.write('\n') 
                s.write('\n')
    with open (filepath, 'r') as f, open (target2, 'w') as t:
        for line in f:
            new_line = ""
            for i in range (2, len(line)):
                new_line += line[i]
            sen = new_line.split('\t')
            t.write(sen[0])
            t.write('\n')
            t.write(sen[1])
            t.write('\n') 
    with open (filepath, 'r') as f, open (target4, 'w') as t, open (target5, 'w') as s:
        for line in f:
            new_line = ""
            for i in range (2, len(line)):
                new_line += line[i]
            sen = new_line.split('\t')
            t.write(sen[0])
            t.write('\n')
            s.write(sen[1])
            s.write('\n') 

readdata('data/valid_none_original.txt', 'valid/valid_personachat.txt',
         'valid/valid_personachat1.txt', 'valid/valid_personachat2.txt',
         'valid/valid_pc_speaker.txt', 'valid/valid_pc_answer.txt')