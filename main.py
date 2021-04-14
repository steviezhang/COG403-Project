import numpy as np
import re
from nltk import Tree
from nltk import induce_pcfg
from nltk import Nonterminal
from nltk.parse.generate import generate

epsilon = 1e-20 
class corpus:
# stores all sentence forms in data
    def __init__(self):
        
        self.sentence_forms = {}
        for i in range(6): # init six levels
            self.sentence_forms[i + 1] = {}
        self.corp = []


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
    return p * np.power(1.0 - p, n - 1, dtype=np.float64)

def compute_prior(G, corpus, n, level, flag=False): # flag for NLTK
    # P : number of productions for grammar G
    # n: number of non terminals for grammar G
    # V: Vocabulary size = # num non terminals + # num terminals = len(corpus[level])
    productions = None
    if flag:
        productions = corpus
    else:
        productions = G
    P = len(productions)
    V = None
    if flag:
        V = len(corpus)
    else:
        V = len(corpus.sentence_forms[level])
    prob_P = np.log(geometric(P, 0.5)+epsilon)
    prob_n = np.log(geometric(n, 0.5)+epsilon)
    log_prior = prob_P + prob_n

    for i in range(P):
        if flag:
            N_i = len(productions[i])
        else:
            N_i = len(list(productions.keys())[i])# num symbols for production i
        prob_N_i = geometric(N_i, 0.5)
        log_prior -= (N_i * np.log(V))
        log_prior += prob_N_i

    return log_prior
    

def compute_log_likelihood(corpus, G, T, level, flag=False):
    # k: number of unique sentence types in corpus
    log_likelihood = 0
    D = None
    k = None
    if flag:
        k = len(corpus)
        D = corpus
    else:
        D = corpus.corp # sentence forms at specified level in corpus
        k = len(D) # get num diff sentence forms at given level
    productions = G
    for i in range(k):
        sl = None
        if flag:
            sl = compute_sentence_likelihood_nltk(productions, D[:50])
        else:
            sentence_i = D[i].split(" ")
            sl = compute_sentence_likelihood(sentence_i, productions)

        if sl != 0:
            log_likelihood += np.log(sl)
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

def compute_sentence_likelihood_nltk(G, productions):
    prob = 0
    prods = list(G.keys())
    S_i = productions
    for p in prods:
        p_split = p.split(" -> ")
        s1 = p_split[0]
        s2 = p_split[1]
        for i, token in enumerate(S_i[:-1]):
            if s1 == token and s2 == S_i[i + 1]:
                prob += np.log(G[p])
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
    prior = compute_prior(adam_levelk_probabilities, data, levelk_nonterminal, k)
    likelihood = compute_log_likelihood(data, adam_levelk_probabilities, PCFG, k)
    logpost = compute_log_posterior(prior, likelihood)

    return prior, likelihood, logpost
    

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

def loadTrees(path):
    with open (path, 'r') as f:
        data = f.read().split("\n\n")

    flattened_data = []
    for i in range(len(data)):
        #flatten it and strip extra whitespace
        flattened_data.append(" ".join(data[i].replace("\n", "").split()))

    tree = []
    for i, s in enumerate(flattened_data[:-2]):
        if "R" in s:
            tree.append(Tree.fromstring(s))

    return tree

def productionsFromTrees(trees):
    productions = []
    for tree in trees:
        productions += tree.productions()
    return productions

def inducePCFGFromProductions(productions):
    S = Nonterminal('S')
    grammar = induce_pcfg(S, productions)
    return grammar


if __name__ == "__main__":

    speakers, struct = read_and_return(directory) # this function was used before perfors sent his data

    corp = []
    types = {}
    for fp in struct:
        for segments in struct[fp]:
            t = ""
            for s in segments[:-1]:
                token = s.split("|")[0].split(":")[0]
                
                if ("#" in token):
                    token = token.split("#")[1]

                t += token + " "
            corp.append(t[:-1])
            splitter = t.split(" ")[:-1]

            for i in range(len(splitter)):
                if (i < (len(splitter) - 1)):
                    tok = splitter[i] + "->" + splitter[i+1]   
            
                    if tok in types:
                        types[tok] += 1
                    else:
                        types[tok] = 1
    
    data = corpus()
    data.sort_sentence_types(types)
    data.corp = corp
    adam_level1 = data.sentence_forms[1] 
    adam_level2 = data.sentence_forms[2]
    adam_level3 = data.sentence_forms[3]
    adam_level4 = data.sentence_forms[4] 
    adam_level5 = data.sentence_forms[5]
    adam_level6 = data.sentence_forms[6]  

    print("FREQUENCY WEIGHTED CFG")
    for i in range(6):
        print("----------------")
        print("LEVEL " + str(i+1))
        prior, likelihood, logpost = test_functions(data.sentence_forms[i+1], i+1)
        print("Log Prior: " + str(prior))
        print("Log Likelihood: " + str(likelihood))
        print("Log Posterior: " + str(logpost))
        
    
    trees = loadTrees("Parsetree/brown-adam.parsed") 
    productions = productionsFromTrees(trees)
    nltkgrammar = inducePCFGFromProductions(productions)

    grammarToParse = str(nltkgrammar).split("\n")
    finalGrammar = []
    grammarDict = {}
    
    for g in grammarToParse:
        finalGrammar.append(g[4:])

    for fg in finalGrammar[1:]:
        gg = fg.split("[")
        rule = gg[0][:-1]
        value = gg[1][:-1]

        grammarDict[rule] = float(value)

    terminal_pattern = "[.?!]"
    terminal_sum = 0
    for j in grammarDict.keys():
        terminal = re.search(terminal_pattern, j)

        if terminal:
            terminal_sum += 1

    print("PROBABALISTIC PCFG")
    prior = compute_prior(grammarDict, productions, terminal_sum, 0, True)
    print("Log Prior: " + str(prior))
    likelihood = compute_log_likelihood(productions, grammarDict, PCFG, 0, True)
    print("Log Likelihood: " + str(likelihood))
    logpost = compute_log_posterior(prior, likelihood)
    print("Log Posterior: " + str(logpost))

    

    

