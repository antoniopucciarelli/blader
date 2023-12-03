#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import customtkinter as ctk
import optimizationGUI

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        minsize = [3000, 1500]

        # customize Ctk 
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # setup event root
        self.ctk_root = self

        # setting up title
        self.title('blader')

        # find the center point
        center_x = 10
        center_y = 10

        # set the position of the window to the center of the screen
        self.geometry(f'{minsize[0]}x{minsize[1]}+{center_x}+{center_y}')

        # setting up grid properties 
        # self.grid_columnconfigure(0, weight=1)
        # self.grid_rowconfigure(0, weight=1)

        self.optimizationFrame = optimizationGUI.OptimizationFrame(master=self)
        self.optimizationFrame.grid(row=0, column=0)

if __name__ == "__main__":

    t = time.time()
    
    app = App()
    
    t = time.time() - t
    print('>>> STARTUP ELAPSED TIME = {0:.2E}'.format(t))
    
    app.mainloop()
