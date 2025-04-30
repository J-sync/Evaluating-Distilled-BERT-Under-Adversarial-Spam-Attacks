
import csv
import string
from collections import defaultdict

def sort_dict_desc(dictionary):
    return dict(sorted(dictionary.items(), key=lambda x: x[1], reverse=True))

def compute_word_frequencies(csv_path="spam.csv"):
    ham_words = defaultdict(int)
    spam_words = defaultdict(int)

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            category, message = row[0], row[1]
            message = message.translate(str.maketrans('', '', string.punctuation)).lower()
            words = message.split()
            word_counts = defaultdict(int)
            for word in words:
                word_counts[word] += 1
            if category == 'ham':
                for word, count in word_counts.items():
                    ham_words[word] += count
            elif category == 'spam':
                for word, count in word_counts.items():
                    spam_words[word] += count

    return sort_dict_desc(ham_words), sort_dict_desc(spam_words)

def main():
    ham_freq, spam_freq = compute_word_frequencies()
    with open("ham_word_freq.csv", "w", newline="") as f_ham, open("spam_word_freq.csv", "w", newline="") as f_spam:
        ham_writer = csv.writer(f_ham)
        spam_writer = csv.writer(f_spam)
        ham_writer.writerow(["word", "frequency"])
        spam_writer.writerow(["word", "frequency"])
        for word, freq in ham_freq.items():
            ham_writer.writerow([word, freq])
        for word, freq in spam_freq.items():
            spam_writer.writerow([word, freq])
    print("Word frequencies saved to 'ham_word_freq.csv' and 'spam_word_freq.csv'.")

if __name__ == "__main__":
    main()
