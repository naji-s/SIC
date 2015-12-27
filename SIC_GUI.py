import pyforms
from   pyforms          import BaseWidget
from   pyforms.Controls import ControlButton
from   pyforms.Controls import ControlFile
from   pyforms.Controls import ControlText

import numpy as np
from SIC_toolkit import SIC_inference, stochastic_SDR_estimator, deterministic_SDR_estimator
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
class SimpleExample1(BaseWidget):
    def __init__(self):
        super(SimpleExample1,self).__init__('SIC GUI')
        #Definition of the forms fields
        self._inference_result = ControlText('Output:', '')
        self._button        = ControlButton('Find causal direction!')
        self._X_loc        = ControlFile('Enter the location of the first time series', '*.txt')
        self._Y_loc        = ControlFile('Enter the location of the second time series', '*.txt')
        #Define the organization of the forms
        self._formset = [ '_X_loc', '_Y_loc', '_button', '_inference_result', ' ']
        #The ' ' is used to indicate that a empty space should be placed at the bottom of the window
        #If you remove the ' ' the forms will occupy the entire window


        #Define the button action
        self._button.value = self.__buttonAction


    def __buttonAction(self):
        """Button action event"""

        #######################################
        # reading the values for X time series
        #######################################

        X = []
        with open(self._X_loc.value) as X_file:
            for line in X_file:
                # [:-2] is to remove \n from the end of each line
                X.append(np.float64(line[:-2]))
        X = np.asarray(X)
        X_file.close()

        #######################################
        # reading the values for Y time series
        #######################################

        Y = []
        with open(self._Y_loc.value) as Y_file:
            for line in Y_file:
                # [:-2] is to remove \n from the end of each line
                Y.append(np.float64(line[:-2]))
        Y = np.asarray(Y)
        Y_file.close()

        #######################################
        # Inferring the causal direction
        #######################################

        output = SIC_inference(X,Y)
        self._inference_result.value = output

#Execute the application
if __name__ == "__main__":   pyforms.startApp( SimpleExample1 )