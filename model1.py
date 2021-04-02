"""
Whats happening? :
(this can all be found in Poverty of Stimulus? paper perfors, tenenbaum)

Building the model framework:
1. First pick some grammar type T from collection {flat, regular, context free}
2. Pick some instance grammar G of type T
3. Generate data D from grammar instance G
4. Compute log(P(G, T|D)) = log(P(D|G, T)) + log(P(G|T)) + Const

To compute the prior probability of G given type T:
1. First specify number of productions P, number of non terminals n, and vocabulary size V
2. Note that P, n, and N_i ~ Geometric(0.5)
3. Computelog(P(G|T)) = log(P(P)) + log(P(n)) + double sum i= 1 -> P, j= 1 -> N_i (log(P(N_i)) - log(V))

To compute likelihood of corpus D given grammar G:
log(P(D|G, T)) = sum i = 1 -> k: log(P(S_i|G, T))


In general, when tetsing the model on three grammar types, the rule of thumb is as sentence complexity
grows, the model corresponding to hierarchical type performs better
"""
import numpy as np

FREE = "Free"
PRG = "Regular"
PCFG = "Context Free"

def geometric(n, p):
    # geometric distribution
    return p * np.pow(1.0 - p, n - 1)

def compute_prior(G, corpus, n):
    # P : number of productions for grammar G
    # n: number of non terminals for grammar G
    # V: Vocabulary size = # num non terminals + # num terminals = len(corpus[level])
    productions = G[level]
    P = len(productions)
    V = len(corpus[level])
    prob_P = np.log(geometric(P, 0.5))
    prob_n = np.log(geometric(n, 0.5))
    log_prior = prob_P + prob_n

    for i in range(P):
        N_i = len(list(productions.keys())[i])# num symbols for production i
        prob_N_i = geometric(N_i, 0.5)
        for j in range(N_i):
            log_prior += prior + np.log(prob_N_i) - np.log(V)
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


if __name__ == "__main__":

    # main compute loop
    # corpus datastruct: read in corpus data from perfors folder and put into corpus class object from process.py
    # grammar table G needs to have probability for each production:
    # G: {keys = levels: values = {keys = production : values = probability of production}}
    # also, you need G for each grammar type in T
    pass
    

