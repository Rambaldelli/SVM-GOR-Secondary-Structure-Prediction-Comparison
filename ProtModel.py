class Prot():   #classe Prot per salvare e standardizare le proteine

    def __init__(self,id, sequence, structure):
        self.id=id
        self.sequence=sequence
        self.structure=structure




class Model():    #classe astratta Model

    def train(self,datat):
        pass

    def predict(self,datap):
        pass

    def GetPred(self):
        pass