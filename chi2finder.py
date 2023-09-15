import numpy as np
from tqdm import tqdm
from framework import AveragedPerdiction

class Chi2Finder:
    def __init__(self, t12_var = 5, iteration = 5, t12_sample = 50, pow_min=-13, pow_max=-10.3, low_resolution = False):
        self.pow        = [pow_min,pow_max]
        self.t12_var    = t12_var
        self.iteration  = iteration
        self.t12_sample = t12_sample
        self.n_sample   = self.t12_sample//self.iteration
        if low_resolution:
            self.muj_array = np.logspace(self.pow[0],self.pow[1],int((self.pow[1]-self.pow[0])*20))
        else:
            self.muj_array = np.logspace(self.pow[0],self.pow[1],int((self.pow[1]-self.pow[0])*55))
            
        self.chi  = np.zeros((len(self.muj_array),self.iteration,self.n_sample))
        self.t12  = np.zeros((len(self.muj_array),self.iteration,self.n_sample))
    def run(self,frame):
        for i,muj in enumerate(tqdm(self.muj_array)):
            t_0   = np.random.uniform(25,35)
            for j in range(self.iteration):
                self.t12[i,j,0] = t_0
                dr_dldt         = frame[[self.t12[i,j,0],muj]]
                pred_bo,pred_su = AveragedPerdiction(dr_dldt,frame.t_e,frame.year,frame.theta,frame.det_su,frame.borom_unoscilated,frame.res,frame.components)
                self.chi[i,j,0] = Chi2(pred_bo, pred_su, frame.data_bo, frame.data_su, frame.f, frame.delta, frame.m12, frame.m12_bar, frame.sig_m12)
                
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
                        dr_dldt             = frame[[self.t12[i,j,ther1],muj]]
                        pred_bo, pred_su    = AveragedPerdiction(dr_dldt,frame.t_e,frame.year,frame.theta,frame.det_su,frame.b8_un,frame.res,frame.components)
                        self.chi[i,j,ther1] = Chi2(pred_bo, pred_su, frame.data_bo, frame.data_su, frame.f, frame.delta, frame.m12, frame.m12_bar, frame.sig_m12)
                        ther1 = ther1 + 1
                ind_min = self.chi[i,j]==np.min(self.chi[i,j])
                t_0     = self.t12[i,j,ind_min][0]
        muj   = self.muj_array[:,np.newaxis,np.newaxis]*np.ones((self.iteration,self.n_sample))
        return np.array([np.ones(self.chi.flatten().shape),self.chi.flatten()/2.,self.t12.flatten(),muj.flatten()])
        
        
def Chi2(pred_bo,pred_su,data_bo,data_su,f,delta,m12,m12_bar,sig_m12):
    #Flux normalization uncertainties taking from solar standard model prediction
    sig_norm_bo = np.array([0.01,0.06,0.01])
    sig_norm_su = 0.12
    
    d_bo = np.array([data_bo['pp']['R'],data_bo['Be7']['R'],data_bo['pep']['R']])
    e_bo = np.array([data_bo['pp']['e'],data_bo['Be7']['e'],data_bo['pep']['e']])
    d_su = data_su[:,2]
    e_su = data_su[:,3]
    
    chi = np.zeros(m12.shape)
    for i in range(m12.shape[0]):
        min_norm_bo = (e_bo**2 + (d_bo*pred_bo[i]*sig_norm_bo**2))/(e_bo**2 + (pred_bo[i]**2*sig_norm_bo**2))

        sig_frac = (sig_norm_su/e_su)**2
        nominator= d_su*pred_su[i]*sig_frac
        denominat= pred_su[i]**2*sig_frac

        min_norm_su = (np.sum(f*nominator[np.newaxis,:],axis=1)+1)/(np.sum(f**2*denominat[np.newaxis,:],axis=1)+1)
        chi_bo      = np.sum((d_bo - min_norm_bo*pred_bo[i])**2/e_bo**2 + (min_norm_bo - 1)**2/sig_norm_bo**2)
        
        
        a = np.ones((f.shape[0],1))*d_su[np.newaxis,:] - f * (min_norm_su[:,np.newaxis] * pred_su[i,np.newaxis,:])
        b = np.ones((f.shape[0],1))*e_su[np.newaxis,:]
        
        
        chi_su = np.sum((a/b)**2,axis=1) + np.sum(delta**2,axis=1)
        
        chi_su = chi_su + ((min_norm_su - 1)/sig_norm_su)**2

        chi[i] = np.min(chi_su) + chi_bo + ((m12[i] - m12_bar)/sig_m12)**2
        
    return np.min(chi)
