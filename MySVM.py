import sklearn.svm 
import ProtModel
import numpy as np
import sys



class SVM(ProtModel.Model):   #classe per modello SVM, estende classe astratta modello generico e utilizza la classe SVC di SkLearn


    def __init__(self, C, G):   
        self.C=C
        self.G=G


    def train(self,datat):  #train the model given a training(labelled) set
        self.datat=datat
        train_set = []
        train_cla=[]
        positions = [x - (17 // 2) for x in range(17)]
        for id in datat:
            for index in range(len(datat[id]['prof'])):
                coded_seq = []
                for p in positions:
                    if index + p < 0 or index + p >= len(datat[id]['prof']):
                        coded_seq.extend(0 for x in range(21))
                    else:
                        coded_seq.extend(datat[id]['prof'][index + p])
                train_set.append(coded_seq)
                train_cla.append(datat[id]['SVMclass'][index])
        #print(self.C,self.gamma)
        #C=self.C
        #gamma=self.gamma
        #print(self.C,self.G)
        mo = sklearn.svm.SVC(C=self.C, kernel='rbf', gamma=self.G)  
        #mo = sklearn.svm.SVC( kernel='rbf')
        #print(mo.C,mo.gamma)
        mo.fit(train_set, train_cla)
        #print('---train----')
        print(mo)
        self.model=mo

    def predicta(self,datap):  #predict an unknown set
        self.datap = datap
        positions = [x - (17 // 2) for x in range(17)]
        self.prediction=[]
        for id in datap:
            test_set = []
            prot=ProtModel.Prot(id,datap[id]['seq'],datap[id]['str'])
            for index in range(len(datap[id]['prof'])):
                coded_seq = []
                for p in positions:
                    if index + p < 0 or index + p >= len(datap[id]['prof']):
                        coded_seq.extend(0 for x in range(21))
                    else:
                        coded_seq.extend(datap[id]['prof'][index + p])
                test_set.append(coded_seq)
            #print('---predict----')
            #print(self.model.C,self.model.gamma)
            #print(self.model)
            predicted_structure = self.model.predict(test_set)
            temp=''
            for p in predicted_structure:
                if p == 1:                            
                    temp=temp+'H'
                elif p== 2:
                    temp=temp+'E'
                elif p==3:
                    temp=temp+'-'
            prot.predicted_structure=temp

            self.prediction.append(prot)

    def GetPred(self):   #return the prediction
        return self.prediction



        #kernel

