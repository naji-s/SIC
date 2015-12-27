import wx
from wx.lib import filebrowsebutton
import sys, os
# a line to add Codes path to be able to use mod_welch and SIC_toolkit module!
libs_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
sys.path.append(libs_path)
from SIC_toolkit import SIC_inference, stochastic_SDR_estimator, deterministic_SDR_estimator
import numpy as np
class ExamplePanel(wx.Panel):
    def __init__(self, parent, size=(300,300)):
        wx.Panel.__init__(self, parent)
        self.quote = wx.StaticText(self, label="Result:", pos=(20, 100))
        self.result = wx.TextCtrl(self, value="", pos=(20, 120))

        # # A multiline TextCtrl - This is here to show how the events work in this program, don't pay too much attention to it
        # self.logger = wx.TextCtrl(self, pos=(300,20), size=(200,300), style=wx.TE_MULTILINE | wx.TE_READONLY)

        # A button
        self.button = wx.Button(self, label="Find the Causal Direction!", pos=(200, 85))
        self.Bind(wx.EVT_BUTTON, self.OnClick,self.button)

        # the edit control - one line version.
        # self.lblname = wx.StaticText(self, label="Your name :", pos=(20,60))
        # self.editname = wx.TextCtrl(self, value="Enter here your name", pos=(150, 60), size=(140,-1))
        # self.Bind(wx.EVT_TEXT, self.EvtText, self.editname)
        # self.Bind(wx.EVT_CHAR, self.EvtChar, self.editname)

        #
        self.X_browse_file = filebrowsebutton.FileBrowseButtonWithHistory(self, labelText='Enter the location for the X time Series:', pos=(10,30))
        self.Y_browse_file = filebrowsebutton.FileBrowseButtonWithHistory(self, labelText='Enter the location for the Y time Series:', pos=(10,50))

        # self.Bind(wx.EVT_BUTTON, self.X_file_read, self.X_browse_file)

        # the combobox Control
        # self.sampleList = ['friends', 'advertising', 'web search', 'Yellow Pages']
        # self.lblhear = wx.StaticText(self, label="How did you hear from us ?", pos=(20, 90))
        # self.edithear = wx.ComboBox(self, pos=(150, 90), size=(95, -1), choices=self.sampleList, style=wx.CB_DROPDOWN)
        # self.Bind(wx.EVT_COMBOBOX, self.EvtComboBox, self.edithear)
        # self.Bind(wx.EVT_TEXT, self.EvtText,self.edithear)

        # # Checkbox
        # self.insure = wx.CheckBox(self, label="Do you want Insured Shipment ?", pos=(20,180))
        # self.Bind(wx.EVT_CHECKBOX, self.EvtCheckBox, self.insure)

        # # Radio Boxes
        # radioList = ['blue', 'red', 'yellow', 'orange', 'green', 'purple', 'navy blue', 'black', 'gray']
        # rb = wx.RadioBox(self, label="What color would you like ?", pos=(20, 210), choices=radioList,  majorDimension=3,
        #                  style=wx.RA_SPECIFY_COLS)
        # self.Bind(wx.EVT_RADIOBOX, self.EvtRadioBox, rb)
    # def X_file_read(self, event=None):
    #     print "salam"

    def OnClick(self,event):
        """Button action event"""

        #######################################
        # reading the values for X time series
        #######################################
        X_loc = self.X_browse_file.GetValue()
        self.X_browse_file.GetHistoryControl().Append(X_loc)
        X = []
        with open(X_loc) as X_file:
            for line in X_file:
                # [:-2] is to remove \n from the end of each line
                X.append(np.float64(line[:-2]))
        X = np.asarray(X)
        X_file.close()

        #######################################
        # reading the values for Y time series
        #######################################

        Y_loc = self.Y_browse_file.GetValue()
        self.Y_browse_file.GetHistoryControl().Append(Y_loc)
        Y = []
        with open(Y_loc) as Y_file:
            for line in Y_file:
                # [:-2] is to remove \n from the end of each line
                Y.append(np.float64(line[:-2]))
        Y = np.asarray(Y)
        Y_file.close()

        #######################################
        # Inferring the causal direction
        #######################################

        output = SIC_inference(X,Y)
        self.result.SetValue(output)

app = wx.App(False)
frame = wx.Frame(None, size=(600,200), title='SIC Algorithm')
panel = ExamplePanel(frame)
frame.Show()
app.MainLoop()