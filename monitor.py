import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class Monitor():
    '''
    A class used to represent the monitoring process.
    '''
    
    def __init__(self, features, risk_scores, batch_size, cumsize, stattest, rmodel, iomodel):
        """
        Parameters
        ----------
        features : pandas dataframe
        risk_scores : series or array
        batch_size : int
            Number of records to provide computations
        cumsize : int
            Number of batches to accumulate before computations (online learning buffer)
        rmodel : class
            Class of the reconstruction model. It must contains methods: getloss(data_batch)
        stattest : function
            The function must take 2 arguments: cummulative batch values, current batch values
        """
        
        # Input Parameters
        self.batch_size = batch_size # number of samples for period
        self.cumsuze = cumsize
        self.features = features # dataset
        self.risk_scores = risk_scores # an array of risk scores <values in 0..1000>
        self.stattest = stattest # Python class with 'test' method which returns statistics and p-value
        self.rmodel = rmodel # PyTorch dumped model OR sklearn extended model
        self.iomodel = iomodel # Autocorrelation model
        
        # Output parameters
        self.time_steps = []
        self.reconstruction_loss = []
        self.statistics = []
        self.pvalues = []
        self.iocorr = []
        
        # Intermediate parameters
        self.prev_risk_scores = []
    
    def fetch_batch(self, features, risk_scores, batch_size):
        n_batches = int(np.ceil(len(features)/batch_size))
        for j in range(n_batches):
            start = j*batch_size
            end = start + batch_size
            features_batch = features[start:end]
            risk_score_batch = risk_scores[start:end]
            yield features_batch, risk_score_batch
    
    def simulation(self):
        for step, batch in enumerate(self.fetch_batch(features=self.features.drop(['part'], axis=1), 
                                                      risk_scores=self.risk_scores, 
                                                      batch_size=self.batch_size)):
            batch_x, batch_y = batch
            reconstruction_loss = self.rmodel.getloss(batch_x) * (1./self.batch_size)
            
            flatten_risk_scores = np.array(self.prev_risk_scores).flatten()
            statistics, p_value = self.stattest(flatten_risk_scores, batch_y)
            
            self.reconstruction_loss.append(float(reconstruction_loss))
            self.statistics.append(statistics)
            self.pvalues.append(p_value)
            self.time_steps.append(step)
            
            if len(self.reconstruction_loss) >= self.cumsuze:
                iocorr_results, p_value_corr = self.iomodel(self.reconstruction_loss[-self.cumsuze:], 
                                                            self.statistics[-self.cumsuze:])
            else:
                iocorr_results = 0
            self.iocorr.append(iocorr_results)
            
            self.prev_risk_scores.append(batch_y)
            if len(self.prev_risk_scores) > self.cumsuze:
                self.prev_risk_scores = self.prev_risk_scores[1:]
        
    def plot(self, type):
        values = {'loss':self.reconstruction_loss, 'pval':self.pvalues, 'stat':self.statistics, 'corr':self.iocorr}[type]
        plt.plot(self.time_steps, values, 'bo')
        plt.plot(pd.Series(values).rolling(window=50).mean(), linewidth=3)
        for vline in np.cumsum(np.array(self.features.groupby('part')['part'].count())):
            plt.axvline(vline/self.batch_size, linestyle='--', color='r')
        plt.xlabel('Time (step * batch)')
        plt.ylabel('{}'.format(type.upper()))
        plt.show()