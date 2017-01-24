import csv
import json
import nltk

# Converter by Sam Bowman (bowman@nyu.edu)

# Converts quora_duplicate_questions.csv, as distributed, to SNLI's formats.
# Data can be found here: https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs
# Note: No parsing yet, just crude tokenization.

# Instructions: Install nltk, move into the same directory as the source file, run.


LABELS = ['entailment', 'neutral']

dev_set = []
test_set = []
training_set = []

with open('quora_duplicate_questions.tsv', 'rbU') as csvfile:
    reader = csv.reader(csvfile, delimiter="\t")
    for i, row in enumerate(reader):
        if i < 1:
            continue
        if len(row[3]) < 1 or len(row[4]) < 1:
            continue

        example = {}
        example['sentence1'] = row[3]
        example['sentence2'] = row[4]
        example['gold_label'] = LABELS[int(row[5])]
        example['pairID'] = row[0]

        example['sentence1_parse'] = example['sentence1_binary_parse'] = ' '.join(nltk.word_tokenize(example['sentence1']))
        example['sentence2_parse'] = example['sentence2_binary_parse'] = ' '.join(nltk.word_tokenize(example['sentence2']))

        if i <= 10000:
            dev_set.append(example)
        elif i <= 20000:
            test_set.append(example)
        else:
            training_set.append(example)

def write_json(data, filename):
    with open(filename, 'w') as outfile:
        for item in data:
            json.dump(item, outfile, sort_keys=True)
            outfile.write('\n')

def write_txt(data, filename):
    with open(filename, 'w') as outfile:
        outfile.write(
            'gold_label\tsentence1_binary_parse\tsentence2_binary_parse\tsentence1_parse\tsentence2_parse\tsentence1\tsentence2\tcaptionID\tpairID\tlabel1\tlabel2\tlabel3\tlabel4\tlabel5\n')
        for item in data:
            tab_sep_string = item['gold_label'] + '\t' + \
                item['sentence1_binary_parse'] + '\t' + item['sentence2_binary_parse'] + '\t' + \
                item['sentence1_parse'] + '\t' + item['sentence2_parse'] + '\t' + \
                item['sentence1'] + '\t' + item['sentence2'] + \
                '\t\t' + item['pairID']
            for i in range(5):
                tab_sep_string += '\t'
            outfile.write(tab_sep_string + '\n')
        outfile.close()

write_json(training_set, 'quora_duplicate_questions_train.jsonl')
write_json(dev_set, 'quora_duplicate_questions_dev.jsonl')
write_json(test_set, 'quora_duplicate_questions_test.jsonl')

write_txt(training_set, 'quora_duplicate_questions_train.txt')
write_txt(dev_set, 'quora_duplicate_questions_dev.txt')
write_txt(test_set, 'quora_duplicate_questions_test.txt')
