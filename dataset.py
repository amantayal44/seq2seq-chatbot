import os
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import re

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Ref : https://www.tensorflow.org/tutorials/text/nmt_with_attention 
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    # removing contractions
    sentence = re.sub(r"i'm","i am",sentence)
    sentence = re.sub(r"let's","let us",sentence)
    sentence = re.sub(r"\'s"," is",sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"n't"," not",sentence)
    sentence = re.sub(r"n'","ng",sentence)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,']+", " ", sentence)
    sentence = sentence.strip()

    return sentence

def load_conversations(lines_filename,conversations_filename,max_samples):

    id2line = {}
    with open(lines_filename,errors='ignore') as files:
        lines = files.readlines()
    for line in lines:
        parts = line.replace('\n','').split(' +++$+++ ')
        id2line[parts[0]] = parts[4]

    questions,answers = [],[]
    with open(conversations_filename,errors='ignore') as files:
        lines = files.readlines()
    for line in lines:
        parts = line.replace('\n','').split(' +++$+++ ')
        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
        for i in range(len(conversation)-1):
            questions.append(preprocess_sentence(id2line[conversation[i]]))
            answers.append(preprocess_sentence(id2line[conversation[i+1]]))
            if len(questions) >= max_samples:
                return questions,answers

    return questions,answers


def tokenize_and_filter(questions,answers,tokenizer,start_token,
    end_token,max_length):
    
    tokenized_questions,tokenized_answers = [],[]
    for (question,answer) in zip(questions,answers):

        sentence1 = start_token + tokenizer.encode(question) + end_token
        sentence2 = start_token + tokenizer.encode(answer) + end_token

        if max(len(sentence1),len(sentence2)) <= max_length:
            tokenized_questions.append(sentence1)
            tokenized_answers.append(sentence2)

    tokenized_questions = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_questions,maxlen=max_length,padding='post')
    tokenized_answers = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_answers,maxlen=max_length,padding='post')

    return tokenized_questions,tokenized_answers

def get_dataset(max_samples,max_length,batch_size,buffer_size = 10000,
    validation_split = None):

    #downloading corpus
    path_to_zip = tf.keras.utils.get_file(
      'cornell_movie_dialogs.zip',
      origin=
      'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
      extract=True)

    path_to_dataset = os.path.join(
        os.path.dirname(path_to_zip),"cornell movie-dialogs corpus")

    lines_filename = os.path.join(path_to_dataset, 'movie_lines.txt')
    conversations_filename = os.path.join(path_to_dataset,
        'movie_conversations.txt')

    questions,answers = load_conversations(lines_filename,
        conversations_filename,max_samples)

    # making subowrd 8K tokenizer
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        questions + answers,target_vocab_size=2**13)

    start_token = [tokenizer.vocab_size]
    end_token = [tokenizer.vocab_size + 1]
    vocab_size = tokenizer.vocab_size + 2

    questions,answers = tokenize_and_filter(questions,answers,tokenizer,
        start_token,end_token,max_length)

    if validation_split is not None:
        questions,test_questions,answers,test_answers = train_test_split(
            questions,answers,test_size= validation_split,random_state=42)

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'encoder_inputs' : questions,
            'decoder_inputs' : answers[:,:-1]
        },answers[:,1:]))
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)


    meta_data = {
    'vocab_size' : vocab_size,
    'start_token' : start_token,
    'end_token' : end_token,
    'max_length' : max_length
    }

    if validation_split is not None: 
        test_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'encoder_inputs' : test_questions,
                'decoder_inputs' : test_answers[:,:-1]
            },test_answers[:,1:]))
        test_dataset = test_dataset.cache()
        test_dataset = test_dataset.batch(batch_size)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset,test_dataset,tokenizer,meta_data

    return dataset,tokenizer,meta_data









