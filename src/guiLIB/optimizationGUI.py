import os
import logging
import matplotlib
import numpy         as     np
import customtkinter as     ctk
from   tkinter       import filedialog
from   geometryLIB   import optimizer

matplotlib.use('TkAgg')

from matplotlib.figure                 import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class OptimizationFrame(ctk.CTkFrame):
    def __init__(self, master: ctk.CTk, logger: logging.Logger):
        # initializing directory
        super().__init__(master)

        # getting working directory
        self.workingDir = os.getcwd()

        # intializing file name 
        self.fileName = None

        # initializing number of DOF for the suction side parametrization
        self.Nsuct = 4

        # initializing number of DOF for the pressure side parametrization
        self.Npress = 4 

        # initializing blade coordinates 
        self.bladeCoords = None

        # setting up optimization properties 
        self.nMax    = 4 
        self.nPoints = 100

        # logger definition
        self.logger = logger

        # setting up frame
        self.setupFrame()

    def setupFrame(
            self,
            corner_radius: int = 10,
            pady:          int = 5
        ) -> None:
        '''
        This function setup the study frame of the app.
        '''

        # setting up label frame
        self.frame = ctk.CTkFrame(
            self,
            width         = 300,
            height        = 100,
            corner_radius = 10,
            # bg_color      = 'transparent'
        )
        self.frame.grid(row=0, column=0, rowspan=10, padx=0, pady=5, sticky='nwe')
    
        # image plotting
        self.loadButton = ctk.CTkButton(master=self.frame, text='LOAD DATA', command=self.getData, corner_radius=corner_radius)
        self.loadButton.grid(row=0, column=0, columnspan=2, pady=pady, padx=5, sticky='we')

        # sliders 
        self.NsuctBox = FloatSpinbox(master=self.frame, logger=self.logger, label='SUCTION', width=270, step_size=1)
        self.NsuctBox.set(4)
        self.NsuctBox.grid(row=1, column=0, columnspan=2, pady=pady, padx=5, sticky='nswe')

        self.NpressBox = FloatSpinbox(master=self.frame, logger=self.logger, label='PRESSURE', width=270, step_size=1)
        self.NpressBox.set(4)
        self.NpressBox.grid(row=2, column=0, columnspan=2, pady=pady, padx=5, sticky='nswe')

        self.deltaAngleBox = FloatSpinbox(master=self.frame, logger=self.logger, label='DELTA ANGLE', width=270, step_size=1, minVal=1, maxVal=10)
        self.deltaAngleBox.set(1)
        self.deltaAngleBox.grid(row=3, column=0, columnspan=2, pady=pady, padx=5, sticky='nswe')

        self.switchBox = SwitchBox(master=self.frame, label='INTERPOLATION', logger=self.logger, values=["linear", "cubic"])
        self.switchBox.grid(row=4, column=0, padx=5, pady=pady, columnspan=2, sticky='nswe')

        # optimization button 
        self.optButton = ctk.CTkButton(master=self.frame, text='OPTIMIZE', command=self.optimize, corner_radius=corner_radius)
        self.optButton.grid(row=5, column=0, columnspan=2, pady=pady, padx=5, sticky='we')

        # save button 
        self.saveButton = ctk.CTkButton(master=self.frame, text='SAVE', command=self.save, corner_radius=corner_radius)
        self.saveButton.grid(row=6, column=0, columnspan=2, pady=pady, padx=5, sticky='we')

        # chart 
        self.plotCoordinates(row=0, column=1)
        
        # logging text
        self.logText(row=0, column=2)

    def optimize(self) -> None:
        '''
        This function optimizes the blade.
        '''

        # logging 
        self.logger.info('>>> optimizing {0} into Kulfan parameters'.format(self.fileName))

        # optimizing blade
        self.blade, self.kulfanParameters, self.bladeData, cost, _, angle = optimizer.optimizeBlade(self.bladeCoords, self.NsuctBox.value, self.NpressBox.value, deltaAngle=self.deltaAngleBox.value, nMax=self.nMax, kind=self.switchBox.segementedButtonVar.get(), nPoints=self.nPoints, plot=False, save=False)

        # normalizing data
        if angle == 0:
            self.bladeData, _, _, _ = optimizer.bladeInOrigin(self.bladeData)
        else:
            maxX           = max(self.bladeData[:,0])
            self.bladeData = self.bladeData / maxX 

            # updating blade properties 
            deltaY = abs(self.bladeData[0,1] - self.bladeData[-1,1]) / 2 - abs(self.bladeCoords[0,1] - self.bladeCoords[-1,1]) / 2
            self.bladeData[:,1] = self.bladeData[:,1] - deltaY 

        try:
            self.bladeLine.set_xdata(self.bladeData[:,0])
            self.bladeLine.set_ydata(self.bladeData[:,1])
        except: 
            self.bladeLine, = self.axes.plot(self.bladeData[:,0], self.bladeData[:,1], 'r', linewidth=3)

        # logging 
        self.logger.info('>>> OPTIMIZATION RESULTS')
        self.logger.info('\t>>> Nsuct = {0:d}'.format(self.NsuctBox.value))
        self.logger.info('\t>>> Npress = {0:d}'.format(self.NpressBox.value))
        self.logger.info('\t>>> DELTA ANGLE = {0:d}'.format(self.deltaAngleBox.value))
        self.logger.info('\t>>> ANGLE = {0:.3f}'.format(angle))
        self.logger.info('\t>>> KIND = {0}'.format(self.switchBox.segementedButtonVar.get()))
        self.logger.info('\t>>> cost = {0:.3E}'.format(cost))
        self.blade.printout(logger=self.logger)
        
        # plottin data
        self.canvas.draw()

    def save(self) -> None:
        '''
        This function allows the app to save the optimized data.
        '''

        # only *.dat files are allowed
        filetypes = [('dat files', '*.dat')]
        
        # getting file name
        fileName = filedialog.asksaveasfilename(filetypes=filetypes, initialfile='data')

        # removing *.dat file
        fileName = fileName[0:-4]
        
        # saving blade properties
        self.blade.save(fileName, self.logger)
        self.logger.info('>>> blade coordinates are saved into: {0:s}*.dat'.format(fileName))

    def getData(self) -> None:
        '''
        This function sets the actions to be made on <LOAD> button click.
        '''

        self.getFileName()

        # check on file name 
        if self.fileName == '':
            self.logger.info('>>> problem getting file name')
            self.logger.info('>>> using last blade coordinates in the system')

        self.loadData()
        self.replot()

    def getFileName(self) -> None:
        '''
        This function get the name of the file which stores the blade coordinates.
        '''

        # only *.dat files are allowed
        filetypes = [('dat files', '*.dat')]

        # calling internal tkinter function for the file pick up
        self.fileName = filedialog.askopenfilename(
            title      = 'Select airfoil coordinates file',
            initialdir = self.workingDir,
            filetypes  = filetypes
        )   

        print('>>> FILENAME = {0}'.format(self.fileName))

    def loadData(self) -> None: 
        '''
        This function loads the blade coordinates from a .dat file.
        '''

        self.logger.info('>>> reading data from {0}'.format(self.fileName))
        with open(self.fileName) as f:
                bladeCoords      = np.loadtxt(f)
                self.bladeCoords = np.array(bladeCoords)
                self.bladeCoords, _, _, _ = optimizer.bladeInOrigin(self.bladeCoords, scale=True)
      
    def plotCoordinates(
            self,
            row:    int,
            column: int
        ) -> None:
        '''
        This function plots the blade coordinates got from a .dat file. 
        '''
        
        # setting up frame
        self.plotFrame = ctk.CTkFrame(master=self)

        # create a figure
        figure = Figure(figsize=(10, 10), dpi=100)

        # create FigureCanvasTkAgg object
        self.canvas = FigureCanvasTkAgg(figure, self.plotFrame)

        # create axes
        axes            = figure.add_subplot()
        self.axes       = axes
        self.line,      = self.axes.plot([], [], 'k', linewidth=3)
        self.bladeLine, = self.axes.plot([], [], 'r', linewidth=4.5)

        # trying plotting data 
        try:
            self.line.set_xdata(self.bladeCoords[:,0])
            self.line.set_ydata(self.bladeCoords[:,1])
        except:
            print('>>> INITIALIZING PLOTTING CHART')
        finally:
            self.axes.set_xlim(-0.1,1.1)
            self.axes.set_ylim(-1,1)
            self.axes.set_aspect('equal')
            self.axes.set_title('Blade')
            self.axes.grid(visible=True, linestyle='dotted')
            self.axes.set_ylabel('y')
            self.axes.set_xlabel('x')

        # setting up tkinter properties
        self.plotFrame.grid(row=row, column=column, padx=5)
        self.canvas.get_tk_widget().grid(row=0, column=1)
        self.canvas.draw()

    def replot(self) -> None:
        '''
        This function replots the blade coordinates.
        '''

        # plotting data 
        try: 
            self.line.set_xdata(self.bladeCoords[:,0])
            self.line.set_ydata(self.bladeCoords[:,1])
        except:
            self.line, = self.axes.plot(self.bladeCoords[:,0], self.bladeCoords[:,1], 'k', linewidth=3)
        
        # redrawing data
        self.axes.set_aspect('equal', adjustable='datalim')
        self.canvas.draw()

        # logging
        self.logger.info('>>> plotting data')

    def logText(self, row: int, column: int) -> None:
        '''
        This function initializes the text box where the logging text has to be printed.
        '''

        # setting up text box
        self.mytext = ctk.CTkTextbox(master=self, width=600, state="disabled")
        self.mytext.grid(row=row, column=column, padx=5, pady=5, sticky='nswe')

class MyHandlerText(logging.StreamHandler):
    def __init__(self, textctrl):
        logging.StreamHandler.__init__(self) # initialize parent
        self.textctrl = textctrl

    def emit(self, record):
        msg = self.format(record)
        self.textctrl.configure(state="normal")
        self.textctrl.insert("end", msg + "\n")
        self.flush()
        self.textctrl.configure(state="disabled")

class SwitchBox(ctk.CTkFrame):
    def __init__(
            self,
            master: ctk.CTkFrame,
            label:  str,
            logger: logging.Logger,
            values: list,
            width:  int = 300,
            height: int = 32
        ) -> None:

        super().__init__(master=master, width=width, height=height)

        self.logger = logger

        self.grid_columnconfigure((0, 2), weight=0) 
        self.grid_columnconfigure(1, weight=1)      

        self.labelInterp = ctk.CTkLabel(self, width=width - 3*height, text=label, bg_color='transparent')
        self.labelInterp.grid(row=0, column=0, padx=0, pady=3, sticky='nw')

        self.segementedButtonVar = ctk.StringVar(value=values[0])
        self.segementedButton = ctk.CTkSegmentedButton(self, values=values, width=width - 2*height, variable=self.segementedButtonVar, bg_color='transparent')
        self.segementedButton.grid(row=0, column=1, padx=(0, 3), pady=3, sticky='nwe')

class FloatSpinbox(ctk.CTkFrame):
    def __init__(
            self,
            master:     ctk.CTkFrame,
            label:      str,
            logger:     logging.Logger,
            defaultVal: int          = 4,
            minVal:     int          = 4,
            maxVal:     int          = 20,
            width:      int          = 300,
            height:     int          = 32,
            step_size:  int or float = 1,
            command                  = None,
        ) -> None:

        super().__init__(master=master, width=width, height=height)
        
        # setting up variables
        self.labelString = label
        self.logger      = logger
        self.step_size   = step_size
        self.command     = command
        self.defaultVal  = defaultVal 
        self.value       = defaultVal
        self.minVal      = minVal 
        self.maxVal      = maxVal

        self.grid_columnconfigure((0, 2), weight=0) 
        self.grid_columnconfigure(1, weight=1)      

        self.label = ctk.CTkLabel(self, width=width - height*3, text=label)
        self.label.grid(row=0, column=0, columnspan=2, padx=10, pady=3, sticky='nwe')

        self.subtract_button = ctk.CTkButton(self, text="-", width=height, height=height, command=self.subtract_button_callback)
        self.subtract_button.grid(row=0, column=2, padx=(3, 0), pady=3)

        self.label = ctk.CTkLabel(self, width=height, height=height)
        self.label.configure(text=str(self.defaultVal))
        self.label.grid(row=0, column=3, padx=3, pady=3, sticky="ew")

        self.add_button = ctk.CTkButton(self, text="+", width=height, height=height, command=self.add_button_callback)
        self.add_button.grid(row=0, column=4, padx=(0, 3), pady=3)

    def add_button_callback(self):
        value = min([int(self.label.cget('text')) + self.step_size, self.maxVal])
        self.value = value
        self.label.configure(text=str(value))
        self.logger.info('>>> increasing DOF ({0}) in {1:s}'.format(value, self.labelString))

    def subtract_button_callback(self):
        value = max([int(self.label.cget('text')) - self.step_size, self.minVal])
        self.value = value
        self.label.configure(text=str(value))
        self.logger.info('>>> decreasing DOF ({0}) in {1:s}'.format(value, self.labelString))

    def get(self) -> int or None:
        return int(self.label.cget('text'))
        
    def set(self, value: int):
        self.label.configure(text=str(value))
        self.value = value