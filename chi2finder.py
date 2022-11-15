import numpy as np
from tqdm import tqdm


class Chi2Finder:
  #T12_var is the variance of \theta_{12}
  #iteration is the number of iteration in order to find best \theta_{12}
  #T12_sample is the number of sampling in each iteration
  #low_resolution is determined the resolution of finding \Delta m_{i}^2
    def __init__(self, t12_var = 5, iteration = 5, t12_sample = 50, low_resolution = False):
        self.t12_var    = t12_var
        self.iteration  = iteration
        self.t12_sample = t12_sample
        self.n_sample   = self.t12_sample//self.iteration
        if low_resolution:
            self.muj_array = np.concatenate((np.logspace(-13,-12,20),
                                             np.logspace(-12,-11,20),
                                             np.logspace(-11,-10.3,10)))
        else:
            self.muj_array = np.concatenate((np.logspace(-13,-12,60),
                                             np.logspace(-12,-11,60),
                                             np.logspace(-11,-10.3,30)))
        self.chi  = np.zeros((len(self.muj_array),self.iteration,self.n_sample))
        self.t12  = np.zeros((len(self.muj_array),self.iteration,self.n_sample))
    def run(self,frame):
        for i,muj in enumerate(tqdm(self.muj_array)):
            t_0   = np.random.uniform(25,35)
            for j in range(self.iteration):
                self.t12[i,j,0] = t_0
                self.chi[i,j,0] = frame[[self.t12[i,j,0],muj]]
                var      = self.t12_var*(5-j)/5
                deltat12 = np.random.normal(0,var,2*self.n_sample)
                ther1    = 1
                ther2    = 0
                while ther1 < self.n_sample :
                    t12_auxi  = self.t12[i,j,ther1-1] + deltat12[np.mod(ther2,2*self.n_sample)]
                    ther2 = ther2 + 1
                    if t12_auxi > 70 or t12_auxi < 10:
                        continue
                    else:
                        self.t12[i,j,ther1] = t12_auxi
                        self.chi[i,j,ther1] = frame[[self.t12[i,j,ther1],muj]]
                        ther1 = ther1 + 1
                ind_min = self.chi[i,j]==np.min(self.chi[i,j])
                t_0      = self.t12[i,j,ind_min][0]
        muj   = self.muj_array[:,np.newaxis,np.newaxis]*np.ones((self.iteration,self.n_sample))
        chain = np.array([np.ones(self.chi.flatten().shape),self.chi.flatten()/2.,self.t12.flatten(),muj.flatten()])

        return     np.savetxt('./Result/chain.txt',chain.T,fmt='   '.join(['%i']+['%1.1f']+['%1.1f']+['%1.3e']),
                              header='weight  -log(liklihood) T12  muj ')
