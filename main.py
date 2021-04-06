import numpy as np
import re
class corpus:
# stores all sentence forms in data
    def __init__(self):
        
        self.sentence_forms = {}
        for i in range(6): # init six levels
            self.sentence_forms[i + 1] = {}


    def sort_sentence_types(self, types):
        for t in types:
            freq = types[t]
            if freq >= 500:
                self.sentence_forms[1][t.rstrip("\n")] = freq # we need to strip these newline characters because they shouldn't count as terminal
                self.sentence_forms[2][t.rstrip("\n")] = freq
                self.sentence_forms[3][t.rstrip("\n")] = freq
                self.sentence_forms[4][t.rstrip("\n")] = freq
                self.sentence_forms[5][t.rstrip("\n")] = freq
                self.sentence_forms[6][t.rstrip("\n")] = freq
            if freq >= 300:
                self.sentence_forms[2][t.rstrip("\n")] = freq
                self.sentence_forms[3][t.rstrip("\n")] = freq
                self.sentence_forms[4][t.rstrip("\n")] = freq
                self.sentence_forms[5][t.rstrip("\n")] = freq
                self.sentence_forms[6][t.rstrip("\n")] = freq
            if freq >= 100:
                self.sentence_forms[3][t.rstrip("\n")] = freq
                self.sentence_forms[4][t.rstrip("\n")] = freq
                self.sentence_forms[5][t.rstrip("\n")] = freq
                self.sentence_forms[6][t.rstrip("\n")] = freq
            if freq >= 50:
                self.sentence_forms[4][t.rstrip("\n")] = freq
                self.sentence_forms[5][t.rstrip("\n")] = freq
                self.sentence_forms[6][t.rstrip("\n")] = freq
            if freq >= 10:
                self.sentence_forms[5][t.rstrip("\n")] = freq
                self.sentence_forms[6][t.rstrip("\n")] = freq

            self.sentence_forms[6][t.rstrip("\n")] = freq

FREE = "Free"
PRG = "Regular"
PCFG = "Context Free"

def geometric(n, p):
    # geometric distribution
    return p * np.power(1.0 - p, n - 1, dtype=np.float64)

def compute_prior(G, corpus, n, level):
    # P : number of productions for grammar G
    # n: number of non terminals for grammar G
    # V: Vocabulary size = # num non terminals + # num terminals = len(corpus[level])
    productions = G
    P = len(productions)
    V = len(corpus.sentence_forms[level])
    prob_P = np.log(geometric(P, 0.5))
    prob_n = np.log(geometric(n, 0.5))
    log_prior = prob_P + prob_n

    for i in range(P):
        N_i = len(list(productions.keys())[i])# num symbols for production i
        prob_N_i = geometric(N_i, 0.5)
        log_prior -= (N_i * np.log(V))
        log_prior += prob_N_i
    return log_prior
    
def compute_log_likelihood(corpus, G, T, level):
    # k: number of unique sentence types in corpus
    log_likelihood = 0
    D = corpus[level] # sentence forms at specified level in corpus
    k = len(D) # get num diff sentence forms at given level
    productions = G[level]
    for i in range(k):
        sentence_i = D[i]
        log_likelihood += np.log(compute_sentence_likelihood(sentence_i, productions))
    return log_likelihood

def compute_sentence_likelihood(S_i, productions):
    # sum of probability of generating S_i under all possible parses
    # productions = "S -> U" # example
    prob = 0
    prods = list(productions.keys())
    for p in prods:
        p_split = p.split("->") # change based on how the prod symbols are seperated
        s1 = p_split[0]
        s2 = p_split[1] # should be only two prod symbols per production
        for i, token in enumerate(S_i[:-1]):
            if s1 == token and s2 == S_i[i + 1]:
                prob += productions[p]
    return prob

def compute_log_posterior(log_prior, log_likelihood):

    return log_prior + log_likelihood + np.log((1.0 / 3.0))

def test_functions(adam_levelk, k):

    terminal_pattern = "[.?!]"
    levelk_terminal = 0
    for j in adam_levelk.keys():
        terminal = re.search(terminal_pattern, j)

        if terminal:
            levelk_terminal += 1

    # #turn grammar into probabilities
    total = sum(adam_levelk.values())
    adam_levelk_probabilities = {}
    for j in adam_levelk.keys():
        adam_levelk_probabilities[j] = adam_levelk[j]/total
  

    levelk_nonterminal = (len(adam_levelk) - levelk_terminal)
    print(compute_prior(adam_levelk_probabilities, data, levelk_nonterminal, k))

import os

directory = "Adam/"
people = ["*MOT", "*URS", "*RIC", "*COL", "*CHI"]
def read_and_return(directory):
    speakers = {}
    struct = {}
    append_next = False
    for file_path in os.listdir(directory):
        with open("Adam/" + file_path, "r") as f:
            speakers[file_path] = []
            struct[file_path] = []
            for line in f:
                split = line.split("  ")
                if append_next and split[0][:4] == "%mor":
                    content = split[0].split("\t")[-1]
                    struct[file_path].append(content.split(" "))
                    
                elif split[0][:4] in people[:-1]:
                    speakers[file_path].append(split)
                    append_next = True
                else:
                    append_next = False
    return speakers, struct


if __name__ == "__main__":
    speakers, struct = read_and_return(directory) # this function was used before perfors sent his data
    types = {}
    for fp in struct:
        for segments in struct[fp]:
            t = "S"
            for s in segments[:-1]:
                token = s.split("|")[0].split(":")[0]
                t += "->" + token
            if t in types:
                types[t] += 1
            else:
                types[t] = 1
    
    data = corpus()
    data.sort_sentence_types(types)
    adam_level1 = data.sentence_forms[1] 
    adam_level2 = data.sentence_forms[2]
    adam_level3 = data.sentence_forms[3]
    adam_level4 = data.sentence_forms[4] 
    adam_level5 = data.sentence_forms[5]
    adam_level6 = data.sentence_forms[6]  
    test_functions(adam_level5, 5)
