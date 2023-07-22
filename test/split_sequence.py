import numpy as np
from collections import defaultdict
import pickle
import torch.nn as nn
import torch

seq = 'MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK'

# seq2 =  'VQERTIFFKDGNYKMVSKEELFTGVPILVELDGDVNHKFSVSGEGEGDAYGKLTLKFIACTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPETRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK'

word_dict = defaultdict(lambda: len(word_dict))

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    # print(sequence)
    words = [word_dict[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]
    return np.array(words)

word = split_sequence(seq, 3)

word = torch.from_numpy(word)

embed_word = nn.Embedding(len(word_dict), 3)

word_vectors = embed_word(word)

print(word_vectors)

# print(split_sequence(seq2, 3))
# print(word_dict)

# dump_dictionary(word_dict, './sequence_dict.pickle')