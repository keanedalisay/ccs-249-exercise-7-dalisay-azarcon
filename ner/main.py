import re, ast

from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.utils import to_categorical

class HiddenMarkovModel:
  def __init__(self):
    self.tag_count = {}
    self.word_tag = {}
    self.states = []
    self.transmission_probs = defaultdict(lambda: defaultdict(lambda: 1e-6))
    self.emission_probs = defaultdict(lambda: defaultdict(lambda: 1e-6))
    

  def train(self, sentences, tagged_sentences):
    for i in range(len(sentences)):
      # if (i > 43000):
      #   print(i)
      split_sentence = sentences[i]
      tagged_sentence = tagged_sentences[i]

      for j in range(len(split_sentence)):
        # if (i > 43000):
        #   print(f'{split_sentence[j]} => {len(split_sentence)}')
        if tagged_sentence[j] not in self.tag_count:
          self.tag_count[tagged_sentence[j]] = 1
        else:
          self.tag_count[tagged_sentence[j]] += 1

        if split_sentence[j] not in self.word_tag:
          self.word_tag[split_sentence[j]] = tagged_sentence[j]

    self._transmission_probabilities(sentences, tagged_sentences)
    self._emission_probabilities(sentences, tagged_sentences)

    for tag in self.tag_count:
      self.states.append(tag)

  def viterbi(self, observations):
    # Initialize Viterbi table
    V = [{}]
    path = {}

    # Initialize the first column of Viterbi table
    for state in self.states:
      second_word_tag = self.word_tag[observations[1]] if self.word_tag.get(observations[1]) else 'O'
      trans_prob = self.transmission_probs[state].get(second_word_tag, 1e-6)
      emit_prob = self.emission_probs[state].get(observations[0], 1e-6)
      # print(f'Transmission Prob: {trans_prob}\n', f'Emission Prob: {emit_prob}')
      V[0][state] = trans_prob * emit_prob
      path[state] = [state]

    # Run Viterbi for t > 0
    for t in range(1, len(observations)):
        V.append({})
        new_path = {}

        for curr_state in self.states:
            max_prob, best_prev_state = max(
                (V[t-1][prev_state] * self.transmission_probs[prev_state].get(curr_state, 1e-6) *
                  self.emission_probs[curr_state].get(observations[t], 1e-6), prev_state)
                for prev_state in self.states
            )
            V[t][curr_state] = max_prob
            new_path[curr_state] = path[best_prev_state] + [curr_state]

        path = new_path

    # Find the best last state
    last_index = len(observations) - 1
    last_word_tag = self.word_tag[observations[last_index]] if self.word_tag.get(observations[last_index]) else 'O'
    max_prob, best_last_state = max(
        (V[len(observations)-1][state] * self.transmission_probs[state].get(last_word_tag, 1e-6), state)
        for state in self.states
    )

    self.states = []

    return path[best_last_state]

  def _transmission_probabilities(self, sentences, tagged_sentences):
    for tag in self.tag_count:
      transmissions = {}
      next_tag_count = {}
      for i in range(len(sentences)): # For each sentence
        tagged_sentence = tagged_sentences[i]
        for j in range(len(sentences[i])): # For each word in the sentence
          word_tag = tagged_sentence[j]
          if (re.match(tag, word_tag)): # If the tag matches
              # Get the next word and its tag
              if (j < len(sentences[i]) - 1):
                next_word_tag = tagged_sentences[i][j+1]
              # next_word_tag = next_word_pos[1]
                next_tag_count[next_word_tag] = next_tag_count.get(next_word_tag, 0) + 1 
                transmissions[tag] = next_tag_count # Add the next tag count to the transmissions dictionary
                # print(f"{ne_word_tag} => {next_ne_word_tag}")
    
      for tag in transmissions:
        # print(f"{tag} => {transmissions[tag]}")
        for next_tag in transmissions[tag]:
          # Calculate the transmission probability
          self.transmission_probs[tag][next_tag] = transmissions[tag][next_tag] / self.tag_count[tag]
  
  def _emission_probabilities(self, sentences, tagged_sentences):
    for tag in self.tag_count:
      emissions = {}
      for i in range(len(sentences)): # For each sentence
        tagged_sentence = tagged_sentences[i]
        for j in range(len(sentences[i])): # For each word in the sentence
          word_tag = tagged_sentence[j]
          if (re.match(tag, word_tag)): 
            word = sentences[i][j]
            if (word not in emissions): # If the word is not in the emissions dictionary
              emissions[word] = 1 # Add the word to the emissions dictionary
            else:
              emissions[word] += 1 # Increment the count of the word in the emissions dictionary
      for word in emissions:
        # print(f"{ne_tag} => {word}: {emissions[word] / ne_tag_count[ne_tag]}")
        # Calculate the emission probability
        self.emission_probs[tag][word] = emissions[word] / self.tag_count[tag]

class RNN:
  def __init__(self, vocab_size, embedding_dim, rnn_units, num_tags):
    self.model = Sequential()
    self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    self.model.add(SimpleRNN(rnn_units, return_sequences=True))
    self.model.add(TimeDistributed(Dense(num_tags, activation='softmax')))
    self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[Precision(), Recall()])

  def train(self, X_train, y_train):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

  def predict(self, X_test):
    return self.model.predict(X_test)

def sentence_tagger(sentences, tags):
  word_count = {}
  tag_count = {}
  tagged_sentences = []

  split_sentences = [sentence.split() for sentence in sentences]

  for i in range(len(split_sentences)):
      split_sentence = split_sentences[i]
      sentence_tags = tags[i]
      tagged_sentence = []
      for j in range(len(split_sentence)):
        if split_sentence[j] not in word_count:
          word_count[split_sentence[j]] = 1
        else:
          word_count[split_sentence[j]] += 1

        if sentence_tags[j] not in tag_count:
          tag_count[sentence_tags[j]] = 1
        else:
          tag_count[sentence_tags[j]] += 1

        if split_sentence[j] not in tagged_sentence:
          tagged_sentence.append((split_sentence[j], sentence_tags[j]))

      tagged_sentences.append(tagged_sentence)

  return word_count, tag_count, tagged_sentences

def main():
  # Uses the Named Entity Recognition (NER) Corpus at Kaggle 
  # Accessible at https://www.kaggle.com/datasets/naseralqaydeh/named-entity-recognition-ner-corpus/data

  df = pd.read_csv("./ner.csv") # Check sentence #47592 as the number of words does not match the number of tags

  sentences = df['Sentence'].to_list()
  split_sentences = [sentence.split() for sentence in sentences] 
  tags = df['Tag'].apply(ast.literal_eval)

  hmm = HiddenMarkovModel()
  hmm.train(split_sentences, tags)

  test_sentence = ['Manila', 'is', 'the', 'capital', 'of', 'the', 'Philippines', '.', 'Maria', 'Villafuerte',',', 'a', 'Filipino', 'citizen', ',', 'is', 'a', 'student', 'at', 'the', 'University', 'of', 'the', 'Philippines']
  test_sentence_labels = ['B-geo', 'O', 'O', 'O', 'O', 'O', 'B-geo', 'O', 'B-per', 'B-per', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-org', 'I-org', 'I-org', 'I-org', 'I-org']

  result = hmm.viterbi(test_sentence)

  print(f"\nTest Sentence: {test_sentence}\n")
  print(f"Predicted Tags: {result}")
  print(f"True Tags: {test_sentence_labels}")

  print(f'\nPrecision: {precision_score(result, test_sentence_labels, average="micro", zero_division=0) * 100}%')
  print(f'Recall: {recall_score(result, test_sentence_labels, average="micro", zero_division=0) * 100}%\n')

  # RNN Model
  wc, tc, tagged_sentences = sentence_tagger(sentences, tags)
  maxlen = max([len(s) for s in tagged_sentences])

  words = list(set(wc.keys()))
  tags = list(set(tc.keys()))

  word2idx = {w: i for i, w in enumerate(words)}
  tag2idx = {t: i for i, t in enumerate(tags)}

  X = [[word2idx[w[0]] for w in s] for s in tagged_sentences]
  X = sequence.pad_sequences(maxlen=maxlen, sequences=X, padding="post",value=len(words) - 1)

  y = [[tag2idx[w[1]] for w in s] for s in tagged_sentences]
  y = sequence.pad_sequences(maxlen=maxlen, sequences=y, padding="post", value=tag2idx["O"])
  y = np.array([to_categorical(i, num_classes=len(tags)) for i in y])
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

  rnn = RNN(vocab_size=len(words), embedding_dim=64, rnn_units=64, num_tags=len(tags))
  rnn.train(X_train, y_train)

  p = rnn.predict(np.array([X_test[0]]))
  p = np.argmax(p, axis=-1)
  y_true = np.argmax(y_test, axis=-1)[0]

  print(f"{'Word':15}{'True':5}\t{'Pred'}")
  print("-"*30)
  for (w, t, pred) in zip(X_test[0], y_true, p[0]):
      print(f"{words[w]:15}{tags[t]}\t{tags[pred]}")

  print(f'Precision: {precision_score(y_true, p[0], average="micro") * 100}%')
  print(f'Recall: {recall_score(y_true, p[0], average="micro") * 100}%')

if __name__ == '__main__':
  main()