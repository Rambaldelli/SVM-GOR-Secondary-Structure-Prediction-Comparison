import json
import glob
from numpy import argmax

dic = {}

with open('blindSet.json', 'w') as D:
    path = 'blindT/dssp/blind_test_dssp/*.dssp'
    files = glob.glob(path)
    for file in files:
        id=file.split('/')[3]
        id=id.split(':')[0]
        f = open(file, 'r')
        F = f.readlines()
        dic[id] = {}
        dic[id]['str']=''
        dic[id]['seq']=''
        for lin in F:
        #dssp
            if (lin[16] == 'H' or lin[16] == 'G' or lin[16] == 'I'):
                dic[id]['str']=dic[id]['str']+'H'
            elif (lin[16] == 'B' or lin[16] == 'E'):
                dic[id]['str']=dic[id]['str']+'E'
            else:
                dic[id]['str']=dic[id]['str']+'-'

        #fasta(sequence)
            if lin[13].islower():
                dic[id]['seq']=dic[id]['seq']+'C'
            else:
                dic[id]['seq']=dic[id]['seq']+lin[13]

    # profile (if pssm missing create profiles one hot)
    # define input string
        data = dic[id]['seq']
    # define universe of possible input values
        alphabet = 'ARNDCQEGHILKMFPSTWYVX'
    # define a mapping of chars to integers
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # integer encode input data
        integer_encoded = [char_to_int[char] for char in data]
    # one hot encode
        onehot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(len(alphabet))]
            letter[value] = 1
            onehot_encoded.append(letter)
        dic[id]['prof'] = onehot_encoded


    # SVM class
        dic[id]['SVMclass'] = []
        for i in dic[id]['str']:
            if i == 'H':
                dic[id]['SVMclass'].append(1)
            elif i == 'E':
                dic[id]['SVMclass'].append(2)
            else:
                dic[id]['SVMclass'].append(3)
    json.dump(dic, D)
