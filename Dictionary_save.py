import json
import glob
from numpy import argmax

dic = {}
with open('project/dir.cla.scope.2.06-stable.txt') as scop:
    scop = scop.readlines()
    with open('datatest.json', 'w') as D:
        path = 'project/dssp/*.dssp'
        files = glob.glob(path)
        for file in files[:10]:
            # dssp
            f = open(file, 'r')
            F = f.readlines()
            dic[F[0].replace('\n', '')] = {}
            dic[F[0].replace('\n', '')]['str'] = F[1].replace('\n', '')
            # fasta
            fasta = F[0].replace('\n', '')
            s = open('project/fasta/' + F[0].replace('\n', '').replace('>', '') + '.fasta', 'r')
            S = s.readlines()
            dic[F[0].replace('\n', '')]['seq'] = S[1].replace('\n', '')
            # profile (if pssm missing create profiles one hot)

            profile = '/home/rambo/LAB2/profiles_clean/' + file.split('/')[-1].split('.')[0] + '.fasta.pssm'
            try:
                with open(profile, 'r') as prof:
                    prof = prof.readlines()
                    a = []
                    for line in prof[1:]:
                        line = line.split('	')
                        line = [x.strip() for x in line]
                        a.append(line)
                    dic[F[0].replace('\n', '')]['prof'] = a
            except:

                # define input string
                data = dic[F[0].replace('\n', '')]['seq']
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
                dic[F[0].replace('\n', '')]['prof'] = onehot_encoded

                #continue
            # scop
            for lines in scop[4:]:
                line = lines.split('\t')
                ide = line[0]
                cla = line[3].split('.')
                if ide == F[0].replace('\n', '').replace('>', ''):
                    dic[F[0].replace('\n', '')]['scope'] = cla
            s.close()
            f.close()

            # cross validation set
            with open('/home/um87/project/cv/test0', 'r') as zero, open('/home/um87/project/cv/test1','r') as uno, open('/home/um87/project/cv/test2','r') as due, open('/home/um87/project/cv/test3', 'r') as tre, open('/home/um87/project/cv/test4', 'r') as quatro:
                zero = zero.read().splitlines()
                uno = uno.read().splitlines()
                due = due.read().splitlines()
                tre = tre.read().splitlines()
                quatro = quatro.read().splitlines()
                #print('--------------------------------')
                #print(zero)
                #print(F[0].replace('\n', '').replace('>', ''))
                #print(dic[F[0].replace('\n', '')])
                if F[0].replace('\n', '').replace('>', '') in zero:
                    dic[F[0].replace('\n', '')]['cv'] = 0
                elif F[0].replace('\n', '').replace('>', '') in uno:
                    dic[F[0].replace('\n', '')]['cv'] = 1
                elif F[0].replace('\n', '').replace('>', '') in due:
                    dic[F[0].replace('\n', '')]['cv'] = 2
                elif F[0].replace('\n', '').replace('>', '') in tre:
                    dic[F[0].replace('\n', '')]['cv'] = 3
                elif F[0].replace('\n', '').replace('>', '') in quatro:
                    dic[F[0].replace('\n', '')]['cv'] = 4
                else:
                    dic[F[0].replace('\n', '')]['cv'] = 'none'
            #print(dic[F[0].replace('\n', '')]['cv'])     #dio criceto continua da qui

            # SVM class
            dic[F[0].replace('\n', '')]['SVMclass'] = []
            for i in dic[F[0].replace('\n', '')]['str']:
                if i == 'H':
                    dic[F[0].replace('\n', '')]['SVMclass'].append(1)
                elif i == 'E':
                    dic[F[0].replace('\n', '')]['SVMclass'].append(2)
                else:
                    dic[F[0].replace('\n', '')]['SVMclass'].append(3)
        json.dump(dic, D)
