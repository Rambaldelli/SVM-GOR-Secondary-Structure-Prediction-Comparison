import MySVM,MyGOR
import numpy as np
import sys
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import json

#OPTION REQUIRED
#cv,type,Gamma,C,datat, datap
#cv= yes/no
#type= SVM/GOR
#Gamma= float
#C= float
#datat= labelled data for cross validation or training
#datap= unlabelled data to be predicted / no (if only for cross validation)

class Predictor:   #classe predittore, puo essere usata sia per cross validation che per predizioni
    cv=None
    Type=None
    Gamma=None
    Ci=None
    labels=None
    datat=None
    datap=None

    tot_Q3=None
    mean_Q3 =None
    err_Q3 = None

    tot_sov=None
    mean_sov = None
    err_sov = None

    tot_acc=None
    mean_acc = None
    err_acc = None

    tot_sen=None
    mean_sen =None
    err_sen = None

    tot_ppv=None
    mean_ppv = None
    err_ppv = None

    tot_mcc=None
    mean_mcc = None
    err_mcc = None


    def __init__(self,cv,type,Ci,Gamma,datat,datap):
        self.labels= ["H", "E", "-"]
        self.cv=cv
        self.type=str(type)
        self.Gamma=float(Gamma)
        self.Ci=float(Ci)
        self.err_acc=np.zeros((3))
        self.err_mcc=np.zeros((3))
        self.err_ppv=np.zeros((3))
        self.err_Q3=0
        self.err_sen=np.zeros((3))
        self.err_sov=0
        with open(datat) as datain:
            self.datat = json.load(datain)
        if datap != 'no':
            with open(datap) as datain:
                self.datap = json.load(datain)


    def run(self):     #controlla se cross validation o predizione ed esegue 
        if self.cv == 'yes':
            self.CrossValidation(self.type,self.datat)
            self.stampa()
        else:
            model=self.CreateModel(self.datat,self.datap)
            self.mean_Q3,self.mean_sov,self.mean_acc,self.mean_sen,self.mean_ppv,self.mean_mcc=self.CMatrix(model)
            self.stampa()







    def CMatrix(self,model):    #Given a model calculate the stat based on the three class.
        acc,sen,ppv,mcc=[],[],[],[]
        Q3 = 0
        label_true=[]
        label_pred=[]
        for prot in model.GetPred():
            label_pred.extend(list(prot.predicted_structure))
            label_true.extend(list(prot.structure))
        ConfusionMatrix=multilabel_confusion_matrix(label_true, label_pred,labels=self.labels)
        for matrix in ConfusionMatrix:
            print(matrix)
            tp = np.float64(matrix[0, 0])
            tn = np.float64(matrix[1, 1])
            fn = np.float64(matrix[1, 0])
            fp = np.float64(matrix[0, 1])
            Q3=Q3+tp
            acc.append((tp+tn)/(tp+tn+fn+fp))
            sen.append(tp/(tp+fn))
            ppv.append(tp/(tp+fp))
            mcc.append(((tp*tn)-(fp*fn))/np.sqrt((tp+fn)*(tp+fp)*(tn+fp)*(tn*fn)))
        Q3=Q3/len(label_true)
        sov=self.Sov(model)
        return Q3,sov,acc,sen,ppv,mcc



    def Sov(self,model): #given a model and a specific SS compute the SOV
        finalsov=0
        for prot in model.GetPred():
            SOV=0
            for label in self.labels:
                tot_osserved=0
                diff_type=0
                datP={}
                datS={}

                fragS=0
                fragP=0

                flagP=0
                flagS=0
                datS[str(fragS)] = set()
                datP[str(fragP)] = set()
                sov=0
                for index in range(len(prot.structure)):      #make fragment
                    if prot.structure[index] == label:
                        flagS= 1
                        datS[str(fragS)].add(index)
                        tot_osserved +=1
                    elif prot.structure[index] != label and flagS == 1:
                        flagS= 0
                        fragS += 1
                        datS[str(fragS)]=set()
                    if prot.predicted_structure[index] == label:
                        flagP=1
                        datP[str(fragP)].add(index)
                    elif prot.predicted_structure[index] != label and flagP == 1:
                        flagP = 0
                        fragP += 1
                        datP[str(fragP)]=set()
                if tot_osserved != 0:
                    for frP in datP:    #compute sov
                        for frS in datS:
                            overlap=set(frP).intersection(frS)
                            overlap=len(overlap)
                            sp=len(frP)
                            ss=len(frS)
                            totlap=sp+ss-overlap
                            if overlap >= 1:
                                correction=min(overlap,ss,sp,totlap)
                                sov=sov+((overlap+correction)/totlap)*ss
                    sov=sov*100/tot_osserved
                    SOV=SOV+sov
                    diff_type +=1
            SOV=SOV / diff_type
            finalsov=finalsov+SOV
        finalsov=finalsov/len(model.GetPred())
        return finalsov




    def CrossValidation(self,type,datat):    #Given a single data set and a type divide it in CV and lunch 5 model
        cv=[0,1,2,3,4]
        self.tot_Q3, self.tot_sov, self.tot_acc, self.tot_sen, self.tot_ppv, self.tot_mcc= [],[],[],[],[],[]
        for x in cv :
            test_set,train_set={},{} 
            for id in list(datat.keys()):  #[310:320]
                if datat[id]['cv']==x:
                    test_set[id]=datat[id]
                else:
                    train_set[id]=datat[id]
            mode=None
            mode=self.CreateModel(train_set,test_set)
            Q3,sov,acc,sen,ppv,mcc=self.CMatrix(mode)
            self.tot_Q3.append(Q3)
            self.tot_sov.append(sov)
            self.tot_acc.append(acc)
            self.tot_sen.append(sen)
            self.tot_ppv.append(ppv)
            self.tot_mcc.append(mcc)
            print('---------------END MODEL--------------')
            

        self.mean_Q3=np.mean(self.tot_Q3)
        self.err_Q3=np.std(self.tot_Q3) / np.sqrt(5)

        self.mean_sov=np.mean(self.tot_sov)
        self.err_sov=np.std(self.tot_sov) / np.sqrt(5)

        self.mean_acc=np.mean(self.tot_acc, axis=0)
        self.err_acc=np.std(self.tot_acc, axis=0) / np.sqrt(5)

        self.mean_sen=np.mean(self.tot_sen, axis=0)
        self.err_sen=np.std(self.tot_sen, axis=0) / np.sqrt(5)

        self.mean_ppv=np.mean(self.tot_ppv, axis=0)
        self.err_ppv=np.std(self.tot_ppv, axis=0) / np.sqrt(5)

        self.mean_mcc=np.mean(self.tot_mcc, axis=0)
        self.err_mcc=np.std(self.tot_mcc, axis=0) / np.sqrt(5)

    def stampa(self):   #stampa i risultati della confusion matrix
        if self.cv == 'no': 
            mea=[]
            mea.append(round(self.mean_Q3,4))
            mea.append(round(self.mean_sov,4))
            for j in range(3):
                mea.append(round(self.mean_acc[j],4))
            for j in range(3):
                mea.append(round(self.mean_sen[j],4))
            for j in range(3):
                mea.append(round(self.mean_ppv[j],4))
            for j in range(3):
                mea.append(round(self.mean_mcc[j],4))

            matrix=[]
            matrix.append(['Q3','sov','H acc', 'E acc', '- acc','H sen', 'E sen', '- sen', 'H ppv', 'E ppv', '- ppv', 'H mcc', 'E mcc', '- mcc'])
            matrix.append(mea)
            s = [[str(e) for e in row] for row in matrix]
            lens = [max(map(len, col)) for col in zip(*s)]
            fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
            table = [fmt.format(*row) for row in s]
            print ('\n'.join(table))

            #print(['Q3','sov','H acc', 'E acc', '- acc','H sen', 'E sen', '- sen','H ppv', 'E ppv', '- ppv','H mcc', 'E mcc', '- mcc'])
            #print(mea)
            #print(tabulate([mea], headers=['Q3','sov','H acc', 'E acc', '- acc','H sen', 'E sen', '- sen','H ppv', 'E ppv', '- ppv','H mcc', 'E mcc', '- mcc']))


        elif self.cv == 'yes':
            mo=[[],[],[],[],[]]
            mea=[]
            er=[]
            for i in range(5):
                mo[i].append(round(i,4))
                mo[i].append(round(self.tot_Q3[i],4))
                mo[i].append(round(self.tot_sov[i],4))
                for j in range(3):
                    mo[i].append(round(self.tot_acc[i][j],4))
                for j in range(3):
                    mo[i].append(round(self.tot_sen[i][j],4))
                for j in range(3):
                    mo[i].append(round(self.tot_ppv[i][j],4))
                for j in range(3):
                    mo[i].append(round(self.tot_mcc[i][j],4))
            mea.append('mean')
            mea.append(round(self.mean_Q3,4))
            mea.append(round(self.mean_sov,4))
            for j in range(3):
                mea.append(round(self.mean_acc[j],4))
            for j in range(3):
                mea.append(round(self.mean_sen[j],4))
            for j in range(3):
                mea.append(round(self.mean_ppv[j],4))
            for j in range(3):
                mea.append(round(self.mean_mcc[j],4))
            er.append('error')
            er.append(round(self.err_Q3,4))
            er.append(round(self.err_sov,4))
            for j in range(3):
                er.append(round(self.err_acc[j],4))
            for j in range(3):
                er.append(round(self.err_sen[j],4))
            for j in range(3):
                er.append(round(self.err_ppv[j],4))
            for j in range(3):
                er.append(round(self.err_mcc[j],4))
            
            matrix=[]
            matrix.append(['Model','Q3','sov','H acc', 'E acc', '- acc','H sen', 'E sen', '- sen','H ppv', 'E ppv', '- ppv','H mcc', 'E mcc', '- mcc'])
            matrix.append(mo[0])
            matrix.append(mo[1])
            matrix.append(mo[2])
            matrix.append(mo[3])
            matrix.append(mo[4])
            matrix.append(mea)
            matrix.append(er)
            s = [[str(e) for e in row] for row in matrix]
            lens = [max(map(len, col)) for col in zip(*s)]
            fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
            table = [fmt.format(*row) for row in s]
            print ('\n'.join(table))

        else:
            print ('andata male')








    def CreateModel(self,train_set,test_set):  #determina il tipo di modello richiesto SVM / GOR e addestra / predice il modello
        if self.type=='GOR':
            mod=MyGOR.GOR()
        elif self.type=='SVM':
            mod=MySVM.SVM(self.Ci,self.Gamma)   #self.C,self.Gamma
        else:
            print('affanculo')
        mod.train(train_set)
        mod.predicta(test_set)
        return mod

if __name__ == "__main__": #main entry point
    modelz=Predictor(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])  #cv,type,Gamma,C,datat, datap
    modelz.run()

#gamma piccolo 
#c 2
