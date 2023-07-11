import numpy as np

'''This class is the basic representation of a material. This will be a tensor that will encode all of the data that
will be input to the model.'''

class Material:


    def __init__(self,a,b,c,alpha,beta,gamma,spacegroup):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.spacegroup = spacegroup
        self.symmetries = np.arange(154)
        self.find_symmetries()

    def find_symmetries(self):
        self.symmetries = np.arange(154)

    def flatten(self):
        1+1