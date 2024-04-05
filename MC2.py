#!/usr/bin/env python
# coding: utf-8

# In[362]:


#Jesse Choi
#This file contains the various types of call and put options -- the values of which are calculated with the monte carlo method

from MC1 import MCStockSimulator
import numpy as np 


# In[363]:


class MCStockOption(MCStockSimulator):
    """encapsulate the idea of a Monte Carlo stock option, and 
    will contain some additional data members( that are not part 
    of class MCStockSimulator and are required to run stock-price 
    simulations and calculate the option’s payoff."""
    
    def __init__(self, s, x, t, r, sigma, nper_per_year, num_trials):
        """Constructor method that initializes input values"""
        super().__init__(s, t, r, sigma, nper_per_year)
        self.x = x
        self.num_trials = num_trials
    
    def __repr__(self):
        """create a nicely formatted printout of the MCStockOption object, 
        which will be useful for debugging your work."""
        
        return f"MCStockOption (s= ${self.s:.2f}, x=${self.x:.2f}, t={self.t:.2f} (years), r={self.mu:.2f}, sigma= {self.sigma:.2f}, nper_per_year= {self.nper_per_year}, num_of_trials={self.num_trials})" 

    def value(self):
        """will return the value of the option. This method cannot be concretely 
        implemented in this base class, but will be overridden in each subclass 
        (see below). For now, the method should print out an informational message 
        and return 0."""
        
        print("Base class MCStockOption has no concrete implementation of .value().")
        return 0
    
    def stderr(self):
        """return the standard error of this option’s value."""
        if 'stdev' in dir(self):
            return self.stdev / np.sqrt(self.num_trials)
        return 0


# In[364]:


class MCEuroCallOption(MCStockOption):
    """Monte carlo call option"""
    
    def __repr__(self):
        """create a nicely formatted printout of the MCEuroCallOption object, 
        which will be useful for debugging your work."""
        
        return f"MCEuroCallOption (s= ${self.s:.2f}, x=${self.x:.2f}, t={self.t:.2f} (years), r={self.mu:.2f}, sigma= {self.sigma:.2f}, nper_per_year= {self.nper_per_year}, num_of_trials={self.num_trials})" 
    
    def value(self):
        """calculates the value of call option"""
        opt_vals = [] #setting up list that will hold all the values for it to be averaged
        
        for i in range(1, self.num_trials+1):  #calculate value for n number of trials 
            
            stock_vals = self.generate_simulated_stock_values()
            value = max(stock_vals[-1] - self.x,0)*(np.exp(-1 * self.mu *self.t))
            opt_vals.append(value)
            
        self.mean = np.mean(opt_vals)
        self.stdev = np.std(opt_vals)

        return self.mean
        
    def stderr(self):
        """calculates standard error of call option values"""
        
        return super().stderr()


# In[365]:


class MCEuroPutOption(MCStockOption):
    """Monte Carlo European Put option"""
    
    def __repr__(self):
        """create a nicely formatted printout of the MCEuroPutOption object, 
        which will be useful for debugging your work."""

        return f"MCEuroPutOption (s= ${self.s:.2f}, x=${self.x:.2f}, t={self.t:.2f} (years), r={self.mu:.2f}, sigma= {self.sigma:.2f}, nper_per_year= {self.nper_per_year}, num_of_trials={self.num_trials})" 

    def value(self):
        """calculates the value of put option"""
        opt_vals = [] #setting up list that will hold all the values for it to be averaged
        
        for i in range(1, self.num_trials+1):  #calculate value for n number of trials 
            stock_vals = self.generate_simulated_stock_values()
            value = max(self.x - stock_vals[-1],0)*(np.exp(-1 * self.mu * self.t))
            opt_vals.append(value)
            
        self.mean = np.mean(opt_vals)
        self.stdev = np.std(opt_vals)

        return self.mean
    
    def stderr(self):
        """calculates standard error of call option values"""
        
        return super().stderr()


# In[366]:


class MCAsianCallOption(MCStockOption):
    """Monte Carlo Asian Call option"""
    
    def __repr__(self):
        """create a nicely formatted printout of the MCAsianCallOption object, 
        which will be useful for debugging your work."""

        return f"MCAsianCallOption (s= ${self.s:.2f}, x=${self.x:.2f}, t={self.t:.2f} (years), r={self.mu:.2f}, sigma= {self.sigma:.2f}, nper_per_year= {self.nper_per_year}, num_of_trials={self.num_trials})" 

    def value(self):
        """calculates the value of call option"""
        opt_vals = [] #setting up list that will hold all the values for it to be averaged
        
        for i in range(1, self.num_trials+1):  #calculate value for n number of trials 
            stock_vals = self.generate_simulated_stock_values()
            value = max(np.mean(stock_vals) - self.x,0)*(np.exp(-1 * self.mu * self.t))
            opt_vals.append(value)
            
        self.mean = np.mean(opt_vals)
        self.stdev = np.std(opt_vals)

        return self.mean
    
    def stderr(self):
        """calculates standard error of call option values"""
        
        return super().stderr()


# In[367]:


class MCAsianPutOption(MCStockOption):
    """Monte Carlo Asian Put Option"""
    
    def __repr__(self):
        """create a nicely formatted printout of the MCAsianPutOption object, 
        which will be useful for debugging your work."""

        return f"MCAsianPutOption (s= ${self.s:.2f}, x=${self.x:.2f}, t={self.t:.2f} (years), r={self.mu:.2f}, sigma= {self.sigma:.2f}, nper_per_year= {self.nper_per_year}, num_of_trials={self.num_trials})" 
    
    def value(self):
        """Calculates the value of put option"""
    
        opt_vals = [] #setting up list that will hold all the values for it to be averaged

        for i in range(1, self.num_trials+1):  #calculate value for n number of trials 
            stock_vals = self.generate_simulated_stock_values()
            value = max(self.x - np.mean(stock_vals),0)*(np.exp(-1 * self.mu * self.t))           
            opt_vals.append(value)

        self.mean = np.mean(opt_vals)
        self.stdev = np.std(opt_vals)

        return self.mean

    def stderr(self):
        """calculates standard error of put option values"""

        return super().stderr()


# In[368]:


class MCLookbackCallOption(MCStockOption):
    """Monte Carlo lookback call option"""
    
    def __repr__(self):
        """create a nicely formatted printout of the MCLookbackCallOption object, 
        which will be useful for debugging your work."""

        return f"MCLookbackCallOption (s= ${self.s:.2f}, x=${self.x:.2f}, t={self.t:.2f} (years), r={self.mu:.2f}, sigma= {self.sigma:.2f}, nper_per_year= {self.nper_per_year}, num_of_trials={self.num_trials})" 
    
    def value(self):
        """Calculates the value of call option"""
    
        opt_vals = [] #setting up list that will hold all the values for it to be averaged

        for i in range(1, self.num_trials+1):  #calculate value for n number of trials 
            stock_vals = self.generate_simulated_stock_values()
            value = max(max(stock_vals) - self.x,0)*(np.exp(-1 * self.mu * self.t))           
            opt_vals.append(value)

        self.mean = np.mean(opt_vals)
        self.stdev = np.std(opt_vals)

        return self.mean

    def stderr(self):
        """calculates standard error of put option values"""

        return super().stderr()


# In[370]:


class MCLookbackPutOption(MCStockOption):
    """Monte Carlo lookback put option"""
    
    def __repr__(self):
        """create a nicely formatted printout of the MCLookbackPutOption object, 
        which will be useful for debugging your work."""

        return f"MCLookbackPutOption (s= ${self.s:.2f}, x=${self.x:.2f}, t={self.t:.2f} (years), r={self.mu:.2f}, sigma= {self.sigma:.2f}, nper_per_year= {self.nper_per_year}, num_of_trials={self.num_trials})" 
    
    def value(self):
        """Calculates the value of put option"""
    
        opt_vals = [] #setting up list that will hold all the values for it to be averaged

        for i in range(1, self.num_trials+1): #calculate value for n number of trials 
            stock_vals = self.generate_simulated_stock_values()
            value = max(self.x - min(stock_vals),0)*(np.exp(-1 * self.mu * self.t))           
            opt_vals.append(value)

        self.mean = np.mean(opt_vals)
        self.stdev = np.std(opt_vals)

        return self.mean

    def stderr(self):
        """calculates standard error of put option values"""

        return super().stderr()

