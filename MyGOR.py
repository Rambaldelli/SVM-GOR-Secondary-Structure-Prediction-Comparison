import ProtModel
from numpy import log10
import numpy as np

class GOR(ProtModel.Model):  #classe per modello GOR, estende classe astratta modello generico
    window=None
    freq = None
    residues = None
    positions = None
    infos = None
    prediction=[]
    model= None
    datap=None
    datat=None





    def __init__(self,w=17):
        self.window=w
        self.prediction=[]
        self.positions = [x - (w // 2) for x in range(w)]
        self.residues = "ARNDCQEGHILKMFPSTWYVX"
        self.freqs ={"#R,H": {x: {str(y): .0 for y in self.positions} for x in self.residues + "-"},
                 "#R,E": {x: {str(y): .0 for y in self.positions} for x in self.residues + "-"},
                 "#R,C": {x: {str(y): .0 for y in self.positions} for x in self.residues + "-"},
                 "#R": {x: {str(y): .0 for y in self.positions} for x in self.residues + "-"}}


        self.infos = {"H": {x: {str(y): .0 for y in self.positions} for x in self.residues},
                 "E": {x: {str(y): .0 for y in self.positions} for x in self.residues},
                 "C": {x: {str(y): .0 for y in self.positions} for x in self.residues}}


    def train(self,datat):    #train the model given a training(labelled) set
        self.datat=datat
        t = 0
        for id in datat:
            p = -1
            for line in datat[id]['prof']:
                p += 1
                t += sum(line)
                for l in range(len(line)):
                    j = self.residues[l]
                    for q in self.positions:
                        if p + q < 0:
                            continue
                        elif p + q >= len(datat[id]['prof']):
                            break
                        i, k = datat[id]['str'][p + q], str(q)
                        if i == "-": i = "C"
                        self.freqs["#R," + i][j][k] += datat[id]['prof'][p + q][l]
                        self.freqs["#R"][j][k] += datat[id]['prof'][p + q][l]

        for i in self.freqs:
            for j in self.freqs[i]:
                if j == "-": break
                for k in self.freqs[i][j]:
                    self.freqs[i][j][k] /= t
            for k in self.freqs[i]["-"]:
                for j in self.residues:
                    self.freqs[i]["-"][k] += self.freqs[i][j][k]
        for i in self.infos:
            for j in self.infos[i]:
                for k in self.infos[i][j]:
                    P_RjS = self.freqs["#R,"+i][j][k]
                    P_S = self.freqs["#R,"+i]["-"][k]
                    P_R = self.freqs["#R"][j][k]
                    if P_RjS == 0 or P_S == 0 or P_R == 0: continue
                    self.infos[i][j][k] = log10(P_RjS/(P_S*P_R))

    def predicta(self,datap):   #predict an unknown set
        self.datap=datap
        struttura=['-','E','H']
        for id in datap:
            prot = ProtModel.Prot(id, self.datap[id]['seq'], self.datap[id]['str'])
            temp=''
            prediction=[]
            for index in range(len(self.datap[id]['prof'])):
                C,H,E=0,0,0
                for res in range(len(self.residues)):
                    for p in self.positions:
                        if p + index < 0:
                            continue
                        elif p + index >= len(self.datap[id]['seq']):
                            break
                        C=C+datap[id]['prof'][index+p][res]*self.infos['C'][self.residues[res]][str(p)]
                        H=H+datap[id]['prof'][index+p][res]*self.infos['H'][self.residues[res]][str(p)]
                        E=E+datap[id]['prof'][index+p][res]*self.infos['E'][self.residues[res]][str(p)]
                list=[C,H,E]
                temp=temp + struttura[list.index(max(list))]

            prot.predicted_structure=temp

            self.prediction.append(prot)
            
    def GetPred(self):  #return the prediction
        return self.prediction

