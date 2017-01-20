import sys
import operator
from collections import defaultdict
from math import log, exp

from nltk.corpus import treebank
from nltk.tag.util import untag  # Untags a tagged sentence. 

unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.

""" Remove trace tokens and tags from the treebank as these are not necessary.
"""
def TreebankNoTraces():
    return [[x for x in sent if x[1] != "-NONE-"] for sent in treebank.tagged_sents()]
        
class BigramHMM:
    def __init__(self):
        self.tag_bigram_counts = defaultdict(float)
        self.tag_unigram_counts = defaultdict(float)
        self.word_tag_bigram_counts = defaultdict(float)
        self.transitions = defaultdict(lambda: float(0.00000001) )
        self.emissions = defaultdict(float )
        self.dictionary = defaultdict( set )
        """ Implement:
        self.transitions, the A matrix of the HMM: a_{ij} = P(t_j | t_i)
        self.emissions, the B matrix of the HMM: b_{ii} = P(w_i | t_i)
        self.dictionary, a dictionary that maps a word to the set of possible tags
        """
    def countGrams(self, training_set):
        for sentence in training_set:
            for (word, tag) in sentence:
                self.tag_unigram_counts[tag] += 1
                self.word_tag_bigram_counts[(word, tag)] += 1
        for sentence in training_set:
            for i in range (0, len(sentence) - 1):
                self.tag_bigram_counts[ (sentence[i][1], sentence[i+1][1] ) ] +=1

    def estimateA(self):
        for (tag1, tag2) in self.tag_bigram_counts:
            self.transitions[ (tag1, tag2) ] = float( self.tag_bigram_counts[ (tag1, tag2) ] ) / float( self.tag_unigram_counts[ (tag1) ] )

    def estimateB(self):
        for (word, tag) in self.word_tag_bigram_counts:
            self.emissions[ (word, tag) ] = float( self.word_tag_bigram_counts[ (word, tag) ] )/ float( self.tag_unigram_counts[ (tag) ] )

    def computeDict(self):
        for (word, tag) in self.word_tag_bigram_counts:
            self.dictionary[word].add(tag)


    def Train(self, training_set):
      """ 
        1. Estimate the A matrix a_{ij} = P(t_j | t_i)
        2. Estimate the B matrix b_{ii} = P(w_i | t_i)
        3. Compute the tag dictionary 
        """
        self.countGrams(training_set)
        self.estimateA()
        self.estimateB()
        self.computeDict()
            
    def ComputePercentAmbiguous(self, data_set):
        unique_tokens = set()
        for sentence in data_set:
            for (word, tag) in sentence:
                unique_tokens.add(word)

        num_ambig = 0
        for word in unique_tokens:
            if len( self.dictionary[word] ) > 1:
                num_ambig += 1

        result = float(num_ambig) /  float( len(unique_tokens) )
        #print "here", num_ambig, len(unique_tokens), result
        return result
        """ Compute the percentage of tokens in data_set that have more than one tag according to self.dictionary. """
        
    def JointProbability(self, sent):

        result = float(0)
        for i in range(1, len(sent) ):
            result = result +  log( self.emissions[( sent[i][0], sent[i][1] ) ] ) + log( self.transitions[ (sent[i-1][1], sent[i][1] ) ]    )
        return exp(result)
        
""" Use Viterbi and predict the most likely tag sequence for every sentence. Return a re-tagged test_set. """
    def Viterbi(self, sent):
        
        trellis = []
        tagged_sent = []
        tagged_sent.append( (start_token, start_token) )
        viterbi = {}
        backpoint= {}
        #initialization
        for tag in self.dictionary[ sent[1][0] ]:
            if ( self.transitions[(start_token, tag)] > 0) and (self.emissions[ (sent[1][0], tag) ] > 0 ) :
                viterbi[(tag, 1)] = log( self.transitions[(start_token, tag)] ) + log( self.emissions[ (sent[1][0], tag) ] )
                backpoint[ (tag, 1) ] = (start_token, start_token)
        #filling trellis
        for T in range(2, len(sent)-1 ):
            obsdict = defaultdict( lambda: -999999999 )
            for S in self.dictionary[ sent[T][0] ]:
                #print S, T
                possvit = []
                possback = {}
                for (prevtag, prevtime) in viterbi:
                    if (prevtime == (T-1) ) and (self.transitions[(prevtag, S)] > float(0) ) and ( self.emissions[ (sent[T][0] , S) ]  > float(0) ):
                        possvit.append(  viterbi[(prevtag, prevtime)] + log(self.transitions[(prevtag, S)] ) + log(self.emissions[(sent[T][0], S )] )    )
                        possback[ (prevtag, prevtime) ] = viterbi[(prevtag, prevtime)] +  log( self.transitions[(prevtag, S)] )            
                if len(possvit) != 0:
                    viterbi[ (S, T) ] = max( possvit )
                if len(possback) != 0:  
                    backpoint[ (S, T) ] = max ( possback, key=possback.get )

        #final transitions to end_token
        T = len(sent)-1
        S = end_token
        possvit = []
        possback = {}
        for (prevtag, prevtime) in viterbi:
            if (prevtime == (T -1) ) and (self.transitions[ (prevtag, S) ] > float(0) ) : 
                possvit.append( viterbi[ (prevtag, prevtime)] + log(self.transitions[ (prevtag, S) ] ) )
                possback[ (prevtag, prevtime) ] = viterbi[ (prevtag, prevtime ) ] + log(self.transitions[ (prevtag, S) ] )
        viterbi[ (S, T) ] = max( possvit )
        backpoint[ (S, T) ] = max( possback, key=possback.get )

        #backpoint
        search = backpoint[ (S, T) ]
        returnsent = [(end_token,end_token)]
        T = T-1
        while (T != start_token):
            returnsent.insert(0, (sent[T][0], search[0]) ) 

            search = backpoint[ search ]
            T = search[1]
        returnsent.insert(0, (start_token, start_token)  )

        return returnsent

    def Test(self, test_set):

        newset = []
        i = 0
        for sentence in test_set:
            i = i+1
            #print "sentence", i
            newsent = self.Viterbi(sentence)
            newset.append( newsent )

        return newset

def MostCommonClassBaseline(training_set, test_set):
    """ Implement the most common class baseline for POS tagging. Return the test set tagged according to this baseline. """
    wordtagcounts = {}

    for sentence in training_set:
        for (word, tag) in sentence:
            if word not in wordtagcounts:
                wordtagcounts[word] = {tag:1}
            elif tag in wordtagcounts[word]:
                wordtagcounts[word][tag] = wordtagcounts[word][tag] + 1
            elif tag not in wordtagcounts[word]:
                wordtagcounts[word][tag] = 1
    
    result_set = []
    for sentence in test_set:
        toadd = []
        for i in range(len(sentence) ):
            toadd.append(  (sentence[i][0], max( wordtagcounts[ sentence[i][0] ], key=wordtagcounts[sentence[i][0] ].get ) ) )
        result_set.append( toadd )
    return result_set

def ComputeAccuracy(test_set, test_set_predicted):
    assert len(test_set) ==len(test_set_predicted)
    totalsent = len(test_set)
    totalwords = 0
    for sentence in test_set:
        totalwords = totalwords + len(sentence) - 2 #exclude boundary tokens

    rightwords = 0
    right_sentences = 0
    for sen_i in range(0, len(test_set) ):
        correct_sentence = True
        for pair_k in range(1, len(test_set[sen_i] ) -1 ): #exclude boundary tokens
            if test_set[sen_i][pair_k] == test_set_predicted[sen_i][pair_k]:
                rightwords +=1
            elif  test_set[sen_i][pair_k] != test_set_predicted[sen_i][pair_k]:
                correct_sentence = False
        if correct_sentence == True:
            right_sentences += 1

    sentence_accuracy = right_sentences * ( pow( totalsent, -1) )
    tag_accuracy = rightwords * ( pow ( totalwords, -1) )

    return (sentence_accuracy, tag_accuracy)

    """ Using the gold standard tags in test_set, compute the sentence and tagging accuracy of test_set_predicted. """
def getvocab(sentence_set):

    vocab = {}
    for sentence in sentence_set:
        for (k, v) in sentence:
            if k not in vocab:
                vocab[k] = 1
            else:
                vocab[k] = vocab[k] + 1
    result = set()
    for word in vocab:
        if vocab[word] > 1:
            result.add(word)
    return result

def PreprocessText(sentence_set, vocabulary):
    for sentence in sentence_set:
        for i in range(len(sentence)):
            if sentence[i][0] not in vocabulary:
                sentence[i] = (unknown_token, sentence[i][1])
    for sentence in sentence_set:
        sentence.insert(0, (start_token, start_token) )
        sentence.append( (end_token, end_token) )

    return sentence_set
def main():
    treebank_tagged_sents = TreebankNoTraces()  # Remove trace tokens. 
    training_set = treebank_tagged_sents[:3000]  # This is the train-test split that we will use. 
    test_set = treebank_tagged_sents[3000:]
    

    vocabulary = getvocab(training_set)
    #print vocabulary
    """ Transform the data sets by eliminating unknown words and adding sentence boundary tokens.
    """

    print " ".join(untag(test_set[0] ) )
    training_set_prep = PreprocessText(training_set, vocabulary)
    test_set_prep = PreprocessText(test_set, vocabulary)
    
    """ Print the first sentence of each data set.
    """
    print training_set[0]
    print test_set_prep[0]
    print " ".join(untag(training_set_prep[0]))  # See nltk.tag.util module.
    print " ".join(untag(test_set_prep[0] ) )

    """ Estimate Bigram HMM from the training set, report level of ambiguity.
    """

    bigram_hmm = BigramHMM()
    bigram_hmm.Train(training_set_prep)


    print "Percent tag ambiguity in training set is %.2f%%." %bigram_hmm.ComputePercentAmbiguous(training_set_prep)
    print "and here are all the tags for the unknown token", bigram_hmm.dictionary[unknown_token]
    print "Joint probability of the first sentence is %s." %bigram_hmm.JointProbability(training_set_prep[0])


    """ Implement the most common class baseline. Report accuracy of the predicted tags.
    """

    test_set_baseline_prediction = MostCommonClassBaseline(training_set_prep, test_set_prep)
    
    basecomp = ComputeAccuracy(test_set_prep, test_set_baseline_prediction)
    print "--- Most common class baseline accuracy ---"
    print "Percent of correct sentences in baseline", basecomp[0]
    print "Percent of correct word taggings in baseline", basecomp[1]

    """ Use the Bigram HMM to predict tags for the test set. Report accuracy of the predicted tags.
    """
    test_set_predicted_bigram_hmm = bigram_hmm.Test(test_set_prep)
    print "--- Bigram HMM accuracy ---"
    finalcomp = ComputeAccuracy(test_set_prep, test_set_predicted_bigram_hmm)    
    print "Percent of correct sentences in final set", finalcomp[0]
    print "Percent of correct word taggings in final set", finalcomp[1]


if __name__ == "__main__": 
    main()

