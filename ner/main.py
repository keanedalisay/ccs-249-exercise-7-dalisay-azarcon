import re, ast

from collections import defaultdict

import pandas as pd

def transmission_probabilities(ne_tag_count, split_sentences, ne_sentences):
  transm_probs = defaultdict(lambda: defaultdict(lambda: 1e-6))

  for ne_tag in ne_tag_count:
    transmissions = {}
    next_ne_tag_count = {}
    for i in range(len(split_sentences)): # For each sentence
      ne_sentence = ne_sentences[i]
      for j in range(len(split_sentences[i])): # For each word in the sentence
        ne_word_tag = ne_sentence[j]
        if (re.match(ne_tag, ne_word_tag)): # If the tag matches
            # Get the next word and its tag
            if (j != len(split_sentences[i]) - 1):
              next_ne_word_tag = ne_sentences[i][j+1]
            # next_word_tag = next_word_pos[1]
            next_ne_tag_count[next_ne_word_tag] = next_ne_tag_count.get(next_ne_word_tag, 0) + 1 
            transmissions[ne_tag] = next_ne_tag_count # Add the next tag count to the transmissions dictionary
            # print(f"{ne_word_tag} => {next_ne_word_tag}")
  
    for ne_tag in transmissions:
      # print(f"{tag} => {transmissions[tag]}")
      for next_ne_tag in transmissions[ne_tag]:
        # Calculate the transmission probability
        transm_probs[ne_tag][next_ne_tag] = transmissions[ne_tag][next_ne_tag] / ne_tag_count[ne_tag]

  return transm_probs

def emission_probabilities(ne_tag_count, split_sentences, ne_sentences):
  emiss_probs = defaultdict(lambda: defaultdict(lambda: 1e-6)) # Default to a small value for unseen words
  for ne_tag in ne_tag_count:
    emissions = {}
    for i in range(len(split_sentences)): # For each sentence
      ne_sentence = ne_sentences[i]
      for j in range(len(split_sentences[i])): # For each word in the sentence
        ne_word_tag = ne_sentence[j]
        if (re.match(ne_tag, ne_word_tag)): 
          word = split_sentences[i][j]
          if (word not in emissions): # If the word is not in the emissions dictionary
            emissions[word] = 1 # Add the word to the emissions dictionary
          else:
            emissions[word] += 1 # Increment the count of the word in the emissions dictionary
    for word in emissions:
      # print(f"{ne_tag} => {word}: {emissions[word] / ne_tag_count[ne_tag]}")
      # Calculate the emission probability
      emiss_probs[ne_tag][word] = emissions[word] / ne_tag_count[ne_tag]

  return emiss_probs


def viterbi(observations, word_tag, ne_tag_count, trans_probs, emit_probs):
  states = [ne_tag for ne_tag in ne_tag_count]

  # Initialize Viterbi table
  V = [{}]
  path = {}

  # Initialize the first column of Viterbi table
  for state in states:
    second_word_tag = word_tag[observations[1]] if word_tag.get(observations[1]) else 'O'
    trans_prob = trans_probs[state].get(second_word_tag, 1e-6)
    emit_prob = emit_probs[state].get(observations[0], 1e-6)
    # print(f'Transmission Prob: {trans_prob}\n', f'Emission Prob: {emit_prob}')
    V[0][state] = trans_prob * emit_prob
    path[state] = [state]

  # Run Viterbi for t > 0
  for t in range(1, len(observations)):
      V.append({})
      new_path = {}

      for curr_state in states:
          max_prob, best_prev_state = max(
              (V[t-1][prev_state] * trans_probs[prev_state].get(curr_state, 1e-6) *
                emit_probs[curr_state].get(observations[t], 1e-6), prev_state)
              for prev_state in states
          )
          V[t][curr_state] = max_prob
          new_path[curr_state] = path[best_prev_state] + [curr_state]

      path = new_path

  # Final transition to END
  last_index = len(observations) - 1
  last_word_tag = word_tag[observations[last_index]] if word_tag.get(observations[last_index]) else 'O'
  max_prob, best_last_state = max(
      (V[len(observations)-1][state] * trans_probs[state].get(last_word_tag, 1e-6), state)
      for state in states
  )

  return path[best_last_state]

def main():
  df = pd.read_csv("./ner.csv")
  df = df[0:100]

  sentences = df['Sentence'].to_list()
  split_sentences = [sentence.split() for sentence in sentences] 
  ne_sentences = df['Tag'].apply(ast.literal_eval)

  word_count = {}
  ne_tag_count = {}
  word_tag = {}

  for i in range(len(split_sentences)):
    split_sentence = split_sentences[i]
    ne_sentence = ne_sentences[i]

    for j in range(len(split_sentence)):
      if split_sentence[j] not in word_count:
        word_count[split_sentence[j]] = 1
      else:
        word_count[split_sentence[j]] += 1

      if (ne_sentence[j] not in ne_tag_count):
        ne_tag_count[ne_sentence[j]] = 1
      else:
        ne_tag_count[ne_sentence[j]] += 1

      if (split_sentence[j] not in word_tag):
        word_tag[split_sentence[j]] = ne_sentence[j]

  transm_probs = transmission_probabilities(ne_tag_count, split_sentences, ne_sentences)
  emiss_probs = emission_probabilities(ne_tag_count, split_sentences, ne_sentences)

  print(viterbi(['George', 'Bush', 'is', 'doing', 'something'], word_tag, ne_tag_count, transm_probs, emiss_probs))


if __name__ == '__main__':
  main()