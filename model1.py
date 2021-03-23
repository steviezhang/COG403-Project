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
import nltk

FREE = "Free"
PRG = "Regular"
PCFG = "Context Free"

def geometric(n, p):
    # geometric distribution
    return p * np.pow(1.0 - p, n - 1)

def compute_prior(P, n, V):
    # P : number of productions
    # n: number of non terminals
    # V: Vocabulary size
    prob_P = np.log(geometric(P, 0.5))
    prob_n = np.log(geometric(n, 0.5))
    log_prior = prob_P + prob_n

    for i in range(P):
        N_i = 1 # num symbols for production i
        prob_N_i = geometric(N_i, 0.5)
        for j in range(N_i):
            log_prior = prior + np.log(prob_N_i) - np.log(V)
    return log_prior
    
def compute_log_likelihood(k):
    # k: number of sentence types in corpus
    log_likelihood = 0
    for i in range(k):
        log_likelihood += np.log(compute_sentence_likelihood())
    return log_likelihood

def compute_log_posterior(log_prior, log_likelihood):

    return log_prior + log_likelihood


def compute_vocab_size(T):
    """TODO (doesnt affect results too much, but it does skew results in the favour
    of linear grammars over context free ones), so for now we just use the literal size
    of the corpus vocab
    """

    if T == FREE:
        pass
    elif T == PRG:
        pass
    elif T == PCFG:
        pass
    else:
        # good enough to return size of vocab for all cases
        pass

if __name__ == "__main__":

    # main compute loop
    # main loop eliminated for now, testing nltk for parsing
    nltk.download("averaged_perceptron_tagger")
    text = nltk.word_tokenize("Lets see if this stuff works")
    print(nltk.pos_tag(text))
    
    

