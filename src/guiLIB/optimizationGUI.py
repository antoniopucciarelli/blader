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
            bg_color      = 'transparent'
        )
        self.frame.grid(row=0, column=0, rowspan=10, padx=0, pady=5, sticky='nwe')
    
        # image plotting
        self.loadButton = ctk.CTkButton(master=self.frame, text='LOAD DATA', command=self.getData, corner_radius=corner_radius)
        self.loadButton.grid(row=0, column=0, pady=pady, padx=5, sticky='we')

        # sliders 
        self.NsuctBox = FloatSpinbox(master=self.frame, label='SUCTION', width=270, step_size=1)
        self.NsuctBox.set(4)
        self.NsuctBox.grid(row=1, column=0, pady=pady, padx=5)

        self.NpressBox = FloatSpinbox(master=self.frame, label='PRESSURE', width=270, step_size=1)
        self.NpressBox.set(4)
        self.NpressBox.grid(row=2, column=0, pady=pady, padx=5)

        # optimization button 
        self.optButton = ctk.CTkButton(master=self.frame, text='OPTIMIZE', command=self.optimize, corner_radius=corner_radius)
        self.optButton.grid(row=3, column=0, pady=pady, padx=5, sticky='we')

        # save button 
        self.saveButton = ctk.CTkButton(master=self.frame, text='SAVE', command=self.save, corner_radius=corner_radius)
        self.saveButton.grid(row=4, column=0, pady=pady, padx=5, sticky='we')

        # chart 
        self.plotCoordinates(row=0, column=1)

    def optimize(self) -> None:
        '''
        This function optimizes the blade.
        '''

        # optimizing blade
        self.blade, self.kulfanParameters, self.bladeData, _, _, _ = optimizer.optimizeBlade(self.bladeCoords, self.Nsuct, self.Npress, nMax=self.nMax, nPoints=self.nPoints, plot=False, save=False)

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

        # only *.dat files are allowed
        filetypes = [('dat files', '*.dat')]
        
        # getting file name
        fileName = filedialog.asksaveasfilename(filetypes=filetypes, initialfile='data')

        # removing *.dat file
        fileName = fileName[0:-4]
        
        # saving blade properties
        self.blade.save(fileName)
        print('>>> BLADE COORDINATES SAVED INTO: {0:s}*.dat'.format(fileName))

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
        axes = figure.add_subplot()
        self.axes = axes
        self.line, = self.axes.plot([], [], 'k', linewidth=3)
        self.bladeLine, = self.axes.plot([], [], 'r', linewidth=3)

        # trying plotting data 
        try:
            self.line.set_xdata(self.bladeCoords[:,0])
            self.line.set_ydata(self.bladeCoords[:,1])
        except:
            print('>>> PROBLEM PRINTING DATA')
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

class FloatSpinbox(ctk.CTkFrame):
    def __init__(
            self,
            master:     ctk.CTkFrame,
            label:      str,
            defaultVal: int          = 4,
            minVal:     int          = 4,
            maxVal:     int          = 20,
            width:      int          = 300,
            height:     int          = 32,
            step_size:  int or float = 1,
            command                  = None,
        ) -> None:

        super().__init__(master=master, width=width, height=height, bg_color='transparent')

        self.step_size  = step_size
        self.command    = command
        self.defaultVal = defaultVal 
        self.minVal     = minVal 
        self.maxVal     = maxVal

        self.grid_columnconfigure((0, 2), weight=0) # buttons don't expand
        self.grid_columnconfigure(1, weight=1)      # entry expands

        self.label = ctk.CTkLabel(self, width=width - height*3, text=label, bg_color='transparent')
        self.label.grid(row=0, column=0, columnspan=2, padx=10, pady=3)

        self.subtract_button = ctk.CTkButton(self, text="-", width=height, height=height, command=self.subtract_button_callback)
        self.subtract_button.grid(row=0, column=2, padx=(3, 0), pady=3)

        self.label = ctk.CTkLabel(self, width=height, height=height)
        self.label.configure(text=str(self.defaultVal))
        self.label.grid(row=0, column=3, padx=3, pady=3, sticky="ew")

        self.add_button = ctk.CTkButton(self, text="+", width=height, height=height, command=self.add_button_callback)
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