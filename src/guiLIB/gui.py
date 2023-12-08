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

        self.optimizationFrame = optimizationGUI.OptimizationFrame(master=self)
        self.optimizationFrame.grid(row=0, column=0)

if __name__ == "__main__":

    t = time.time()
    
    app = App()
    
    t = time.time() - t
    print('>>> STARTUP ELAPSED TIME = {0:.2E}'.format(t))
    
    app.mainloop()