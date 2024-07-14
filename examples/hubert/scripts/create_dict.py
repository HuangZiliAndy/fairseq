import os
import argparse

parser = argparse.ArgumentParser(description='Create dictionary file')
parser.add_argument('text_file', type=str, help='text file')
parser.add_argument('output_file', type=str, help='output file')
args = parser.parse_args()

def main():
    with open(args.text_file, 'r') as fh:
        content = fh.readlines()
    char2cnt = {}
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        for c in line_split:
            if c not in char2cnt:
                char2cnt[c] = 0
            char2cnt[c] += 1
    char_list = [[k, char2cnt[k]] for k in char2cnt.keys()]
    char_list = sorted(char_list, key=lambda x: -x[1])
    with open(args.output_file, 'w') as fh:
        for i in range(len(char_list)):
            fh.write("{} {}\n".format(char_list[i][0], char_list[i][1]))
    return 0

if __name__ == '__main__':
    main()
