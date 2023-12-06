import os
import matplotlib
import numpy         as     np
import customtkinter as     ctk
from   tkinter       import filedialog
from   geometryLIB   import optimizer

matplotlib.use('TkAgg')

from matplotlib.figure                 import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class OptimizationFrame(ctk.CTkFrame):
    def __init__(self, master: ctk.CTk):
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

        # setting up frame
        self.setupFrame(row=0, col=0)

    def setupFrame(
            self, 
            row: int, 
            col: int, 
        ) -> None:
        '''
        This function setup the study frame of the app.
        '''

        # setting up label frame
        self.frame = ctk.CTkFrame(self, corner_radius=10) 
        self.frame['text'] = 'Airfoil'
        self.frame.grid(row=row, column=col, rowspan=10, padx=5, pady=5, sticky='nwe')
    
        # image plotting
        self.loadButton = ctk.CTkButton(master=self.frame, text='LOAD DATA', command=self.getData)
        self.loadButton.grid(row=0, column=0, sticky="ew", pady=3)

        self.plotCoordinates(row=0, column=1)

        # sliders 
        self.NsuctBox = FloatSpinbox(master=self.frame, label='SUCTION', width=150, step_size=1)
        self.NsuctBox.set(4)
        self.NsuctBox.grid(row=1, column=0)

        self.NpressBox = FloatSpinbox(master=self.frame, label='PRESSURE', width=150, step_size=1)
        self.NpressBox.set(4)
        self.NpressBox.grid(row=2, column=0)

        # optimization button 
        self.optButton = ctk.CTkButton(master=self.frame, text='OPTIMIZE', command=self.optimize)
        self.optButton.grid(row=3, column=0, sticky="ew", pady=3)

        # save button 
        self.saveButton = ctk.CTkButton(master=self.frame, text='SAVE', command=self.save)
        self.saveButton.grid(row=4, column=0, sticky="ew", pady=3)

    def optimize(self) -> None:
        '''
        This function optimizes the blade.
        '''

        # optimizing blade
        blade, kulfanParameters, self.bladeData, cost, fig, angle = optimizer.optimizeBlade(self.bladeCoords, self.Nsuct, self.Npress, nMax=self.nMax, nPoints=self.nPoints, plot=False, save=False)

        # normalizing data
        self.bladeData, _, _, _ = optimizer.bladeInOrigin(self.bladeData)

        # updating blade properties 
        deltaY = (self.bladeData[0,1] - self.bladeData[-1,1]) / 2 - (self.bladeCoords[0,1] - self.bladeCoords[-1,1]) / 2
        self.bladeData[:,1] = self.bladeData[:,1] - deltaY 

        try:
            self.bladeLine.set_xdata(self.bladeData[:,0])
            self.bladeLine.set_ydata(self.bladeData[:,1])
        except: 
            self.bladeLine, = self.axes.plot(self.bladeData[:,0], self.bladeData[:,1], 'r', linewidth=3)

        self.canvas.draw()

    def save(self) -> None:
        '''
        This function allows the app to save the optimized data.
        '''
        pass

    def getData(self) -> None:
        '''
        This function sets the actions to be made on <LOAD> button click.
        '''

        self.getFileName()
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

        print('>>> READING DATA FROM {0}'.format(self.fileName))
        with open(self.fileName) as f:
                bladeCoords = np.loadtxt(f)
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
        
        self.plotFrame = ctk.CTkFrame(master=self)

        # create a figure
        figure = Figure(figsize=(10, 10), dpi=100)

        # create FigureCanvasTkAgg object
        self.canvas = FigureCanvasTkAgg(figure, self.plotFrame)

        # create axes
        axes = figure.add_subplot()
        self.axes = axes

        # trying plotting data 
        try:
            self.line, = self.axes.plot(self.bladeCoords[:,0], self.bladeCoords[:,1], 'k', linewidth=3)
            self.bladeLine, = self.axes.plot([], [], 'r', linewidht=3)
        except:
            print('>>> PROBLEM PRINTING DATA')
        finally:
            self.axes.set_aspect('equal', adjustable='datalim')
            self.axes.set_title('Blade')
            self.axes.grid(visible=True, linestyle='dotted')
            self.axes.set_ylabel('y')
            self.axes.set_xlabel('x')

        # setting up tkinter properties
        self.plotFrame.grid(row=row, column=column)
        self.canvas.get_tk_widget().grid(row=0, column=0)
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

class FloatSpinbox(ctk.CTkFrame):
    def __init__(
            self,
            master:     ctk.CTkFrame,
            label:      str,
            defaultVal: int          = 4,
            minVal:     int          = 4,
            maxVal:     int          = 20,
            width:      int          = 100,
            height:     int          = 32,
            step_size:  int or float = 1,
            command                  = None,
        ) -> None:

        super().__init__(master=master, width=width, height=height)

        self.step_size  = step_size
        self.command    = command
        self.defaultVal = defaultVal 
        self.minVal     = minVal 
        self.maxVal     = maxVal

        self.grid_columnconfigure((0, 2), weight=0) # buttons don't expand
        self.grid_columnconfigure(1, weight=1)      # entry expands

        self.label = ctk.CTkLabel(self, width=width, text=label, bg_color='transparent')
        self.label.grid(row=0, column=0, columnspan=2, padx=10, pady=3)

        self.subtract_button = ctk.CTkButton(self, text="-", width=height-6, height=height-6, command=self.subtract_button_callback)
        self.subtract_button.grid(row=0, column=2, padx=(3, 0), pady=3)

        self.label = ctk.CTkLabel(self, width=width-(3*height), height=height-6)
        self.label.configure(text=str(self.defaultVal))
        self.label.grid(row=0, column=3, padx=3, pady=3, sticky="ew")

        self.add_button = ctk.CTkButton(self, text="+", width=height-6, height=height-6, command=self.add_button_callback)
        self.add_button.grid(row=0, column=4, padx=(0, 3), pady=3)

    def add_button_callback(self):
        value = min([int(self.label.cget('text')) + self.step_size, self.maxVal])
        self.label.configure(text=str(value))

    def subtract_button_callback(self):
        value = max([int(self.label.cget('text')) - self.step_size, self.minVal])
        self.label.configure(text=str(value))

    def get(self) -> int or None:
        return int(self.label.cget('text'))
        
    def set(self, value: int):
        self.label.configure(text=str(value))