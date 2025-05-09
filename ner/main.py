import re, ast

from collections import defaultdict

import pandas as pd

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

def main():
  # Uses the Named Entity Recognition (NER) Corpus at Kaggle 
  # Accessible at https://www.kaggle.com/datasets/naseralqaydeh/named-entity-recognition-ner-corpus/data

  df = pd.read_csv("./ner.csv") # Check sentence #47592 as the number of words does not match the number of tags

  sentences = df['Sentence'].to_list()
  split_sentences = [sentence.split() for sentence in sentences] 
  ne_sentences = df['Tag'].apply(ast.literal_eval)

  hmm = HiddenMarkovModel()
  hmm.train(split_sentences, ne_sentences)

  test_sentence = ['Manila', 'is', 'the', 'capital', 'of', 'the', 'Philippines']
  test_sentence_labels = ['B-geo', 'O', 'O', 'O', 'O', 'O', 'B-geo']
  result = hmm.viterbi(test_sentence)

  print(f"Test Sentence: {test_sentence}")
  print(f"Predicted Tags: {result}")

if __name__ == '__main__':
  main()