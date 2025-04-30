
import tensorflow as tf
import tensorflow_hub as hub
import nltk
from nltk.corpus import gutenberg
import pandas as pd
import csv
from random import sample
import random
import os

nltk.download('gutenberg')
nltk.download('punkt')

word_list = gutenberg.words()

# Load the BERT model
bert_model = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3")

def generate_attack_set_bert(email_text, num_words_to_replace=3):
    email_tokens = ["[CLS]"] + nltk.word_tokenize(email_text.lower()) + ["[SEP]"]

    # Fallback if empty
    if not email_tokens:
        return email_text, 0.0

    # Generate input tensors
    input_ids = tf.constant(bert_model.tokenize(email_tokens))[tf.newaxis, :]
    input_mask = tf.ones_like(input_ids)

    # Get embedding
    email_embedding = bert_model([input_ids, input_mask])[0][0]

    email_words = nltk.word_tokenize(email_text.lower())
    words_to_replace = [word for word in email_words if word.isalpha()]
    if len(words_to_replace) == 0:
        return email_text, 1.0

    words_to_replace = sample(words_to_replace, min(num_words_to_replace, len(words_to_replace)))

    potential_replacements = [word for word in word_list if len(word) == len(words_to_replace[0]) and word != words_to_replace[0]]
    if not potential_replacements:
        return email_text, 1.0

    attack_set = email_text.lower()
    for word in words_to_replace:
        replacement_word = sample(potential_replacements, 1)[0]
        attack_set = attack_set.replace(word, replacement_word)

    attack_tokens = ["[CLS]"] + nltk.word_tokenize(attack_set.lower()) + ["[SEP]"]
    attack_input_ids = tf.constant(bert_model.tokenize(attack_tokens))[tf.newaxis, :]
    attack_input_mask = tf.ones_like(attack_input_ids)
    attack_embedding = bert_model([attack_input_ids, attack_input_mask])[0][0]

    similarity = tf.keras.losses.cosine_similarity(email_embedding, attack_embedding)

    return attack_set, similarity.numpy()[0]

def pick_random_spam_email(csv_path="spam.csv"):
    df = pd.read_csv(csv_path)
    spam_df = df[df["category"] == "spam"]
    if spam_df.empty:
        return None
    return spam_df.sample()["email"].values[0]

def main():
    attack_emails = []
    for i in range(30):
        spam_email = pick_random_spam_email()
        if not spam_email:
            continue
        num_words = max(1, int(len(spam_email.split()) * 0.1))
        attack_email, sim_score = generate_attack_set_bert(spam_email, num_words)
        if sim_score > -0.8:  # cosine_similarity in TF returns negative cosine distance
            attack_emails.append((attack_email, round(1 + sim_score, 3)))  # Normalize

    with open("attack.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["attack_email", "similarity"])
        writer.writerows(attack_emails)

if __name__ == "__main__":
    main()
