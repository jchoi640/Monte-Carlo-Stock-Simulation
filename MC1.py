#!/usr/bin/env python
# coding: utf-8

# In[74]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:07:11 2024

@author: jessechoi
"""
#Jesse Choi
# This file contains the base class that simulates monte carlo stock 


import numpy as np
import matplotlib.pyplot as plt


class MCStockSimulator:
    """encapsulates the data and methods required to simulate stock returns 
    and values. This class will serve as a base class for option pricing 
    classes (in part 2).
    """
    def __init__(self, s, t, mu, sigma, nper_per_year):
        """constructor method that initializes input values"""
        self.s = s
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.nper_per_year = nper_per_year
        
    def __repr__(self):
        """create a nicely formatted printout of the MCStockOption object, which will be useful for debugging your work."""
        return f"MCStockSimulator (s= ${self.s:.2f}, t={self.t:.2f} (years), mu= {self.mu:.2f}, sigma= {self.sigma:.2f}, nper_per_year= {self.nper_per_year})"  
    
    def generate_simulated_stock_returns(self):
        """generate and return a np.array (numpy array) containing a sequence 
        of simulated stock returns over the time period t.
        """
        dt = 1/self.nper_per_year
        simulated_return = (self.mu - self.sigma ** 2 /2) * dt + np.random.normal(0, 1,int(self.nper_per_year*self.t))*self.sigma*np.sqrt(dt)
        return simulated_return
        
    def generate_simulated_stock_values(self):
        """will generate and return a np.array (numpy array) containing a 
        sequence of stock values, corresponding to a random sequence of 
        stock return. 
        """
        returns = self.generate_simulated_stock_returns()
        values = np.array([self.s])
        stock_price = self.s
        
        for val in returns: #loop over returns 
            stock_price = stock_price * np.exp(val)
            values = np.append(values, stock_price)
        
        return values
    
    def plot_simulated_stock_values(self, num_trials = 1):
        """will generate a plot of of num_trials series of simulated stock 
        returns. num_trials is an optional parameter; if it is not supplied, 
        the default value of 1 will be used.
        """
        plt.title(f"{num_trials} simulated trials")
        plt.xlabel('years')
        plt.ylabel('values')
        years = np.linspace(0, self.t, self.nper_per_year*self.t+1) #Because we need to include the start price, I had to get creative and use linspace to specifically state that I need nper_per_year*t + 1

        for i in range(num_trials): #loop however many trials there are
            values = self.generate_simulated_stock_values()
            plt.plot(years, values)

