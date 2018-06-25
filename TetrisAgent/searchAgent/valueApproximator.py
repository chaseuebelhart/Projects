from abc import ABC, abstractmethod

class ValueApproximator:
    '''Abstract value approximator class'''

    @abstractmethod
    def predictValue(self, state):
        '''Returns the predicted value of the state'''
        pass
