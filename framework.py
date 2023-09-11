import numpy as np

class FrameWork(object):
    def __init__(self, su_nbin=23, mumi = 'mum2', t12 = 33.4, m12=7.54e-5, m12_nuisance=True):
        #Fermi constant :  1.166 \times 10^{-11}/Mev^2
        self.f_c    = 1.166
        
        #Electron mass : 0.511 Mev
        self.m_e    = 0.511
        
        #h_{bar}c : 1.97 10^{-11} Mev.cm
        self.hbarc = 1.97
        
        #proton mass :  1.67 \times 10^{-27} kg
        self.m_p    = 1.67 
        
        #Nutrino Flux normalization :    Arxiv : 1611.09867 (HZ)
        self.norm  = {'pp' : [5.98],
                      'Be7': [0.9*4.93e-1,0.1*4.93e-1],
                      'pep': [1.44e-2], 
                      'B8' : [5.46e-4]}   #\times 10^{10}
        
        #Neutrino production point weight function : http://www.sns.ias.edu/~jnb/
        load_phi   = np.loadtxt('./Solar_Standard_Model/bs2005agsopflux1.txt', unpack = True)
        self.phi   = {'pp' : load_phi[5,:],
                      'Be7': load_phi[10,:],
                      'pep': load_phi[11,:],
                      'B8' : load_phi[6,:]}
        
        #Electron density inside sun 10^{23}/cm^{3}
        self.n_e  = 6*10**load_phi[2,:]
        
        #Neutrino energy spectrum : http://www.sns.ias.edu/~jnb/
        spectrumB8      = np.loadtxt('./Spectrum/B8_spectrum.txt')
        spectrumpp      = np.loadtxt('./Spectrum/pp_spectrum.txt')
        spectrumbe71    = np.loadtxt('./Spectrum/be71_spectrum.txt')
        spectrumbe72    = np.loadtxt('./Spectrum/be72_spectrum.txt')
        spectrumpep     = np.loadtxt('./Spectrum/pep_spectrum.txt')
        self.spec       = {'pp' : [spectrumpp[:,1]],
                           'Be7': [spectrumbe71[:,1],spectrumbe72[:,1]],
                           'pep': [spectrumpep[:,1]]  ,
                           'B8' : [spectrumB8[:,1]]}
        
        #Neutrino energy in Mev
        self.e_nu       = {'pp' : [spectrumpp[:,0]],
                           'Be7': [spectrumbe71[:,0],spectrumbe72[:,0]],
                           'pep': [spectrumpep[:,0]]),
                           'B8' : [spectrumB8[:,0]]}
        
        #electron recoil energy in Mev
        self.t_e        = {'pp'  : [IntegralLimit(spectrumpp[:,0])],
                           'Be7' : [IntegralLimit(spectrumbe71[:,0]),IntegralLimit(spectrumbe72[:,0])],
                           'pep' : [IntegralLimit(spectrumpep[:,0])],
                           'B8'  : [IntegralLimit(spectrumB8[:,0])]}
        
        #Borexino Data event (count per day per 100 ton) : https://doi.org/10.1038/s41586-018-0624-y 
        #Eur. Phys. J. C 80, 1091 (2020)
        self.data_bo    = {'pp' : {'R' : 134.,'e' : 14.},
                           'Be7': {'R' : 48.3,'e' : 1.3},
                           'pep': {'R' : 2.43,'e' : 0.42}}

        #Super-K Data event (count per year per  kilo ton) :
        self.data_su  = np.loadtxt('./Data/B8_Data_2020.txt')[:su_nbin,:]

        #detector normalization 
        #Borexino : per 100 ton :  3.307 \times 10^{31} 
        #Super-K  : per Kton    :  (10/18) \times 10^{6}/m_p
        self.det_bo = 0.03307              
        self.det_su = (10/18) * 1/self.m_p
        
        #event per day  : 24 \times 60 \times 60 
        self.time   = 24.*6.*6.     
        self.l,self.a,self.theta,self.h   = SunEarthDistance()
        self.year   = 60*60*24*365.25
        
        #electron recoil cross section
        g        = self.hbarc * self.f_c
        self.cs  = {'e'      : {'pp' : [DCS(g,self.m_e,self.e_nu['pp'][i:],t,1) for i,t in enumerate (self.t_e['pp'])], 
                                'Be7': [DCS(g,self.m_e,self.e_nu['Be7'][i],self.t_e['Be7'][:,i],1) for i in range(2)],  
                                'pep':  DCS(g,self.m_e,self.e_nu['pep'][0],self.t_e['pep'],1),
                                'B8' : [DCS(g,self.m_e,self.e_nu['B8'][i:],t,1) for i,t in enumerate (self.t_e['B8'])]}, 
                    'mu/tau' : {'pp' : [DCS(g,self.m_e,self.e_nu['pp'][i:],t,-1) for i,t in enumerate (self.t_e['pp'])], 
                                'Be7': [DCS(g,self.m_e,self.e_nu['Be7'][i],self.t_e['Be7'][:,i],-1) for i in range(2)],  
                                'pep':  DCS(g,self.m_e,self.e_nu['pep'][0],self.t_e['pep'],-1),
                                'B8' : [DCS(g,self.m_e,self.e_nu['B8'][i:],t,-1) for i,t in enumerate (self.t_e['B8'])]}}
        
        #Super-k detector response function   
        self.res  = ResSu(self.data_su,self.t_e['B8'])
                
        shape = np.loadtxt('./Correlated_Errors/Nutrino_shape_Systematic_Uncertainties.txt')[:su_nbin,:]        
        scale = np.loadtxt('./Correlated_Errors/Energy_Scale_Systematic_Uncertainties.txt')[:su_nbin,:]
        resol = np.loadtxt('./Correlated_Errors/Energy_Resolution_Systematic_Uncertainties.txt')[:su_nbin,:]

        self.delta = np.random.normal(0,1,(500,3))

        self.f     = ((1/(1+self.delta[:,0,np.newaxis]*shape[:,1]/100))*
                      (1/(1+self.delta[:,1,np.newaxis]*scale[:,1]/100))*
                      (1/(1+self.delta[:,2,np.newaxis]*resol[:,1]/100)))
        
        #Based on KamLAND
        self.m12_bar  = 7.54e-5
        self.sig_m12  = 5.0e-6
        if m12_nuisance:
            self.m12 = np.linspace(self.m12_bar - (2*self.sig_m12), self.m12_bar + (2*self.sig_m12),7)
        else:
            self.m12 = np.array([m12])    
        self.mumi    = mumi
        self.param   = {'T12' : t12 ,
                        'T13' : 8.57, 
                        'mum1': 0. ,
                        'mum2': 0. ,
                        'mum3': 0. ,
                        'M12' : m12 }
        
        self.pred_bo = np.zeros((self.m12.shape[0],3))
        self.pred_su = np.zeros((self.m12.shape[0],self.data_su.shape[0]))
        
    def __getitem__(self,param_ubdate):
        self.param['T12']     = param_ubdate[0]
        self.param[self.mumi] = param_ubdate[1]
        
        for i in range(self.m12.shape[0]):
            self.param['M12']= self.m12[i]
            pep_pee,pep_pes = SurvivalProbablity(self.phi['pep'], self.e_nu['pep'], self.n_e, self.f_c, self.hbarc, self.param, self.l)
            be7_pee,be7_pes = SurvivalProbablity(self.phi['Be7'], self.e_nu['Be7'], self.n_e, self.f_c, self.hbarc, self.param, self.l)
            pp_pee,pp_pes   = SurvivalProbablity(self.phi['pp'] , self.e_nu['pp'] , self.n_e, self.f_c, self.hbarc, self.param, self.l)
            b8_pee,b8_pes   = SurvivalProbablity(self.phi['B8'] , self.e_nu['B8'] , self.n_e, self.f_c, self.hbarc, self.param, self.l)

            pee = {'pp' : pp_pee, 'Be7' : be7_pee, 'pep' : pep_pee, 'B8' : b8_pee} 
            pes = {'pp' : pp_pes, 'Be7' : be7_pes, 'pep' : pep_pes, 'B8' : b8_pes}
            
            r_pep,r_be7,r_pp,r_b8 = EventRateMaker(pee, pes, self.cs, self.e_nu, self.t_e, self.l, self.spec, self.theta, self.data_su, self.res)

            self.pred_bo[i] = (self.time/self.year) * self.det_bo * (self.a**2/self.h) * np.array([self.norm['pp'] * r_pp, self.norm['Be7'] * r_be7, self.norm['pep'] * r_pep])
            self.pred_su[i] = 365.*(self.time/self.year) * self.det_su * (self.a**2/self.h)*(self.norm['B8']*r_b8/self.data_su[:,-1])
        return Chi2(self.pred_bo, self.pred_su, self.data_bo, self.data_su, self.f, self.delta, self.m12, self.m12_bar, self.sig_m12)

def SunEarthDistance(resolution=0.08):
    a     = (1.521e11 + 1.471e11)/2  #L = 1.5e11 meter 
    e     = (1.521e11 - 1.471e11)/(1.521e11 + 1.471e11)

    theta = np.arange(0,2*np.pi,resolution)
    cos   = np.cos(theta)
    l     = a*(1-e**2)/(1+e*cos)    
    h     = np.trapz(l**2,theta)/(60*60*24*365.25)
    return l,a,theta,h

def IntegralLimit(e,lowt=-4,num=100):
    mint = np.min(e)
    maxt = np.max(e)
    mint = np.log10(mint/(1+m_e/(2*mint)))
    return np.concatenate((np.logspace(lowt,mint,num),e[1:]/(1+m_e/(2*e[1:]))))
        
def DCS(g, m_e, e_nu, t_e, i=1):
    #dsigma/dT_e as function of T_e and E_nu (electron recoil and neutrino energy)

    #weak mixing angle = 0.22342 : https://pdg.lbl.gov/2019/reviews/rpp2019-rev-standard-model.pdf
    sw    = 0.2315

    rho   = 1.0126
    x     = np.sqrt(1 + 2*m_e/t_e)
    it    = (1/6) * ((1/3) + (3 - x**2) * ((x/2) * (np.log(x+1) - np.log(x-1)) - 1))
    if i == 1:
        kappa = 0.9791 + 0.0097 * it
        gl    = rho * (0.5 - kappa * sw) - 1
    if i == -1:
        kappa = 0.9970 - 0.00037 * it
        gl    = rho * (0.5 - kappa * sw)
    gr    = -rho * kappa * sw
    
    z     = t_e/e_nu
    #radiative correction
    ap  = 1/(137*np.pi)
    fm  = 0
    fp  = 0
    fmp = 0
    
    a1  = gl**2 * (1 + ap * fm)
    a2  = gr**2 * (1 + ap * fp) * (1-z)**2
    a3  = gr * gl * (1 + ap * fmp) * (m_e/e_nu) * z
    
    return  2 * g**2 * (m_e/np.pi) * (a1 + a2 - a3) * 10 #\times 10^{-45} in cm^2

def ResSu(data, t_e):
    r   = np.zeros((data.shape[0],t_e.shape[0]))
    for j in range (data.shape[0]):
        e_nu = np.linspace(data[j,0],data[j,1])
        for i,t in enumerate(t_e):
            sig  = (-0.084+0.349*np.sqrt(t)+0.04*t)
            a    = (1/(np.sqrt(2*np.pi)*sig))*np.exp(-0.5*(t-e_nu)**2/sig**2)
            r[j,i] = np.trapz(a,e_nu)
    return r

def SurvivalProbablity(phi, enu, n_e, f_c, hbarc, param, ls): 
    pel   = np.zeros((ls.shape[0],enu.shape[0]))
    psl   = np.zeros((ls.shape[0],enu.shape[0]))
    
    util= np.ones((n_e.shape[0],enu.shape[0]))
    ne  = np.reshape(n_e ,(n_e.shape[0],1))*util
    e   = np.reshape(enu ,(1,enu.shape[0]))*util

    ve  = 2 * np.sqrt(2) * f_c * ne * hbarc**3 * 1e-9 * e
    den = np.sqrt((param['M12'] * np.cos(2*(np.pi/180) * param['T12'])- ve)**2 + (param['M12'] * np.sin(2*(np.pi/180) * param['T12']))**2)         
    nom = param['M12'] * np.cos(2*(np.pi/180) * param['T12']) - ve
    tm  = 0.5*np.arccos(nom/den)

    sin = np.sin((np.pi/180) * param['T12'])**2 * np.cos((np.pi/180) * param['T13'])**4
    cos = np.cos((np.pi/180) * param['T12'])**2 * np.cos((np.pi/180) * param['T13'])**4

    for j,l in enumerate(ls):
        ae1 = cos * np.cos(tm)**2  * np.cos(10*param['mum1']*l/(hbarc*2*e))**2
        ae2 = sin * np.sin(tm)**2  * np.cos(10*param['mum2']*l/(hbarc*2*e))**2
        ae3 = np.sin((np.pi/180)*param['T13'])**4 * np.cos(10*param['mum3']*l/(hbarc*2*e))**2

        pee = ae1 + ae2 + ae3
        pel[j]  = np.sum(np.reshape(phi,(n_e.shape[0],1))*pee,axis=0)

        as1 = np.cos((np.pi/180) * param['T13'])**2 * np.cos(tm)**2  * np.sin(10*param['mum1']*l/(hbarc*2*e))**2
        as2 = np.cos((np.pi/180) * param['T13'])**2 * np.sin(tm)**2  * np.sin(10*param['mum2']*l/(hbarc*2*e))**2
        as3 = np.sin((np.pi/180) * param['T13'])**2 * np.sin(10*param['mum3']*l/(hbarc*2*e))**2

        pes = as1 + as2 + as3
        psl[j]  = np.sum(np.reshape(phi,(n_e.shape[0],1))*pes,axis=0)

    return pel, psl

def EventRateMaker(pee, pes, cs, enu, t_e, l, spec, theta, data_su, res):
    rpep_l = np.zeros(l.shape[0])
    
    rbe7_l = np.zeros((l.shape[0],2))
    
    rpp_t  = np.zeros((l.shape[0],enu['pp'].shape[0]))
    rpp_l  = np.zeros(l.shape[0])
    
    rb8_t  = np.zeros((l.shape[0],t_e['B8'].shape[0]))
    rb8_l  = np.zeros((l.shape[0],data_su.shape[0]))
    rb8    = np.zeros((data_su.shape[0]))
    
    rpep_l = np.trapz((cs['e']['pep']*pee['pep'][:,0,np.newaxis] + cs['mu/tau']['pep']*(1-pee['pep'][:,0,np.newaxis]-pes['pep'][:,0,np.newaxis])),t_e['pep'],axis=1)
    rpep   = np.trapz(rpep_l,theta)
    
    for i in range(2):
        rbe7_l[:,i] = np.trapz((cs['e']['Be7'][i]*pee['Be7'][:,i,np.newaxis] + cs['mu/tau']['Be7'][i]*(1-pee['Be7'][:,i,np.newaxis]-pes['Be7'][:,i,np.newaxis])),t_e['Be7'][:,i],axis=1)
                                                        
    rbe7 =  (0.1*np.trapz(rbe7_l[:,0],theta) + np.trapz(rbe7_l[:,1],theta))

    for i,t in enumerate(t_e['pp']):
        rpp_t[:,i] = np.trapz(spec['pp'][np.newaxis,i:]* (cs['e']['pp'][i][np.newaxis,:]*pee['pp'][:,i:] + cs['mu/tau']['pp'][i][np.newaxis,:]*(1-pee['pp'][:,i:]-pes['pp'][:,i:])),enu['pp'][i:],axis=1)
                                                 
    rpp_l = np.trapz(rpp_t,t_e['pp'],axis=1)  
    rpp   = np.trapz(rpp_l,theta)

    for i,t in enumerate(t_e['B8']):
        cs1 = cs['e']['B8'][i][np.newaxis,:]
        cs2 = cs['mu/tau']['B8'][i][np.newaxis,:]
        rb8_t[:,i] = np.trapz(spec['B8'][i:]*(cs1*pee['B8'][:,i:] + cs2*(1-pee['B8'][:,i:]-pes['B8'][:,i:])),enu['B8'][i:])
                                              
    for k in range(data_su.shape[0]):
        rb8_l[:,k] = np.trapz(res[k]*rb8_t,t_e['B8'],axis=1)

    rb8  =  np.trapz(rb8_l,theta,axis=0)
    return rpep,rbe7,rpp,rb8


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
