#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import logging
import customtkinter as ctk
import optimizationGUI

module_logger = logging.getLogger(__name__)

class App(ctk.CTk):
    def __init__(self, logger: logging.Logger):
        super().__init__()

        # customize Ctk 
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # setup event root
        self.ctk_root = self

        # setting up title
        self.title('blader')

        # setting up logger 
        self.logger = logger

        self.optimizationFrame = optimizationGUI.OptimizationFrame(master=self, logger=self.logger)
        self.optimizationFrame.grid(row=0, column=0)

if __name__ == "__main__":

    t = time.time()
    
    app = App(logger=module_logger)
    
    t = time.time() - t

    stderrHandler = logging.StreamHandler()  # no arguments => stderr
    module_logger.addHandler(stderrHandler)
    guiHandler = optimizationGUI.MyHandlerText(app.optimizationFrame.mytext)
    module_logger.addHandler(guiHandler)
    module_logger.setLevel(logging.INFO)
    module_logger.info('>>> startup elapsed time = {0:.2E} s'.format(t)) 

    app.mainloop()