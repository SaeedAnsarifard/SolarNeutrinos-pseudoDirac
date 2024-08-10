import numpy as np

f_c    = 1.166 #Fermi constant :  1.166 \times 10^{-11}/Mev^2
m_e    = 0.511 #Electron mass : 0.511 Mev
hbarc  = 1.97 #h_{bar}c : 1.97 10^{-11} Mev.cm
m_p    = 1.67 #proton mass :  1.67 \times 10^{-27} kg
g      = hbarc * f_c
    
class FrameWork(object):
    #su_nbin is the number of electron recoil spectrum bins in the SuperKamiokande experiment (Max is 23)
    #mumi stands for \Delta m_i^2 which is a Majorana mass splitting sourced by pseudo-Dirac scenario
    #t12 stands for \theta_{12}
    #m12 stands for \Delta m_{12}^2
    #m12_nuisance whether be considered as a nuisance parameter around or fixed to the value of KamLAND
    
    def __init__(self, su_nbin=23, mumi = 'mum2', t12 = 33.4, m12=7.54e-5, m12_nuisance=True):
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
                           'pep': [spectrumpep[:,0]],
                           'B8' : [spectrumB8[:,0]]}
        
        #electron recoil energy in Mev
        self.uppt       = 100
        self.t_e        = {'pp'  : [IntegralLimit(spectrumpp[:,0],m_e,uppt=self.uppt)],
                           'Be7' : [IntegralLimit(spectrumbe71[:,0],m_e,uppt=self.uppt),
                                    IntegralLimit(spectrumbe72[:,0],m_e,uppt=self.uppt)],
                           'pep' : [IntegralLimit(spectrumpep[:,0],m_e,uppt=self.uppt)],
                           'B8'  : [IntegralLimit(spectrumB8[:,0],m_e,uppt=self.uppt)]}
        
        #Borexino Data event (count per day per 100 ton) : https://doi.org/10.1038/s41586-018-0624-y 
        #Eur. Phys. J. C 80, 1091 (2020)
        self.data_bo    = {'pp' : {'R' : 134.,'e' : 14.},
                           'Be7': {'R' : 48.3,'e' : 1.3},
                           'pep': {'R' : 2.43,'e' : 0.42}}

        #Super-K Data event (count per year per  kilo ton) :
        self.su_nbin  = su_nbin
        self.data_su  = np.loadtxt('./Data/B8_Data_2020.txt')[:self.su_nbin,:]
        
        #geometric charactrestic : resolution of delta theta, theta is the angle between earth and sun
        self.resolution                 = 0.08
        self.l,self.a,self.theta,self.h = SunEarthDistance(self.resolution)
        self.year                       = 60*60*24*365.25
    
        #Super-k detector response function   
        self.res  = ResSu(self.data_su,self.t_e['B8'][0])
                
        shape = np.loadtxt('./Correlated_Errors/Nutrino_shape_Systematic_Uncertainties.txt')[:self.su_nbin,:]
        scale = np.loadtxt('./Correlated_Errors/Energy_Scale_Systematic_Uncertainties.txt')[:self.su_nbin,:]
        resol = np.loadtxt('./Correlated_Errors/Energy_Resolution_Systematic_Uncertainties.txt')[:self.su_nbin,:]

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
        
        #Unoscilated signal is produced to compare with the SuperKamiokande results. For more info see their papers!
        #Super-K  : number of target per kilo ton    :  (10/18) \times 10^{6}/m_p -> 3.3 \times 10^{32} 
        self.det_su = 365.25 * 24. * 6. * 6. * (10/18) * 1/m_p #number of target in kilo ton in a year times 10^{35}
        #B8 phi SNO : 5.25e \times 10^6 cm^2 s^-1
        borom_spec,borom_total = BoromUnoscilated(self.t_e['B8'][0],self.e_nu['B8'][0],self.spec['B8'][0],g,m_e,self.uppt,self.data_su,self.res)
        self.borom_unoscilated_spectrum = self.det_su * (2 * np.pi/self.year) * 5.25e-4 * (self.a**2/self.h) * borom_spec
        self.borom_unoscilated_total    = self.det_su * (2 * np.pi/self.year) * 5.25e-4 * (self.a**2/self.h) * borom_total
        
        self.dr_dldt    = [{'pp' :[[]] , 'Be7' :[[],[]] , 'pep' :[[]] , 'B8' :[[]] } for i in range(self.m12.shape[0])]
        self.components = ['pp','Be7','pep','B8']
        
    def __getitem__(self,param_ubdate):
        self.param['T12']     = param_ubdate[0]
        self.param[self.mumi] = param_ubdate[1]
        for i in range(self.m12.shape[0]):
            self.param['M12']= self.m12[i]
            pee        = {'pp' :[[]] , 'Be7' :[[],[]] , 'pep' :[[]] , 'B8' :[[]] }
            pes        = {'pp' :[[]] , 'Be7' :[[],[]] , 'pep' :[[]] , 'B8' :[[]] }
            for c in self.components:
                for j in range(len(self.t_e[c])):
                    t = self.t_e[c][j]
                    e = self.e_nu[c][j]
                    sp= self.spec[c][j]
                    pee[c][j],pes[c][j] = SurvivalProbablity(self.phi[c], e, self.n_e, f_c, hbarc, self.param, self.l)
                    r = np.zeros((self.l.shape[0],t.shape[0]))
                    k = 0
                    for z,ts in enumerate(t):
                        if z<=self.uppt:
                            cse    = DCS(g,m_e,e,ts,1)
                            csmu   = DCS(g,m_e,e,ts,-1)
                            r[:,z] = np.trapz(sp*(cse*pee[c][j]+csmu*(1-pee[c][j]-pes[c][j])),e,axis=1)
                        else:
                            cse    = DCS(g,m_e,e[k:],ts,1)
                            csmu   = DCS(g,m_e,e[k:],ts,-1)
                            r[:,z] = np.trapz(sp[k:]*(cse*pee[c][j][:,k:]+csmu*(1-pee[c][j][:,k:]-pes[c][j][:,k:])),e[k:],axis=1)
                            k      = k + 1
                    self.dr_dldt[i][c][j] = (self.a**2/self.h) * self.norm[c][j] * r #number of event per each delta theta per electron recoil times 10^{-35}
        return self.dr_dldt

def SunEarthDistance(resolution=0.08):
    a     = (1.521e11 + 1.471e11)/2  #L = 1.5e11 meter 
    e     = (1.521e11 - 1.471e11)/(1.521e11 + 1.471e11)

    theta = np.arange(0,2*np.pi,resolution)
    cos   = np.cos(theta)
    l     = a*(1-e**2)/(1+e*cos)    
    h     = np.trapz(l**2,theta)/(60*60*24*365.25)
    return l,a,theta,h

def IntegralLimit(e,m_e,lowt=-4,uppt=100):
    mint = np.min(e)
    maxt = np.max(e)
    mint = np.log10(mint/(1+m_e/(2*mint)))
    return np.concatenate((np.logspace(lowt,mint,uppt),e[1:]/(1+m_e/(2*e[1:]))))
        
def DCS(g, m_e, e_nu, t_e, i=1):
    #dsigma/dT_e as function of T_e and E_nu (electron recoil and neutrino energy)

    #weak mixing angle = 0.22342 : https://pdg.lbl.gov/2019/reviews/rpp2019-rev-standard-model.pdf
    sw    = 0.2315

    #Bahcall, John N., Marc Kamionkowski, and Alberto Sirlin. 
    #"Solar neutrinos: Radiative corrections in neutrino-electron scattering experiments." 
    #Physical Review D 51.11 (1995): 6146.
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
    #radiative correction we dont consider it currently
    ap  = 1/(137*np.pi)
    fm  = 0
    fp  = 0
    fmp = 0
    
    a1  = gl**2 * (1 + ap * fm)
    a2  = gr**2 * (1 + ap * fp) * (1-z)**2
    a3  = gr * gl * (1 + ap * fmp) * (m_e/e_nu) * z
    
    return  2 * g**2 * (m_e/np.pi) * (a1 + a2 - a3) * 10 #\times 10^{-45} in cm^2

def ResSu(data, t_e):
    #PhysRevD.109.092001
    r   = np.zeros((data.shape[0],t_e.shape[0]))
    for j in range (data.shape[0]):
        e_nu = np.linspace(data[j,0],data[j,1])
        for i,t in enumerate(t_e):
            sig  = (-0.05525+0.3162*np.sqrt(t)+0.04572*t)
            a    = (1/(np.sqrt(2*np.pi)*sig))*np.exp(-0.5*(t-e_nu)**2/sig**2)
            r[j,i] = np.trapz(a,e_nu)
    return r
    
    
def ResBo(t_e, n_thl = 150, n_thu = 428):
    #the thresholds are coresponding to DOI:10.1016/j.astropartphys.2022.102778
    n     = np.arange(n_thl,n_thu,1)
    #Based on doi:10.1007/JHEP07(2022)138 [arXiv:2204.03011 [hep-ph]]
    nb    = -8.065244 + 493.2560*t_e - 64.09629*t_e**2 + 4.146102*t_e**3
    sigma = 1.21974 + 1.31394*np.sqrt(nb) - 0.0148585*nb
    rt    = np.zeros((len(n),len(t_e)))
    for i in range(len(t_e)):
        rt[:,i]     = (1/(sigma[i]*np.sqrt(2*np.pi)))*np.exp(-0.5*((n-nb[i])/sigma[i])**2)
    return rt

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

def BoromUnoscilated(t,e,sp,g,m_e,uppt,data_su,res):
    r         = np.zeros(t.shape)
    num_event = np.zeros(len(data_su))
    k         = 0
    for z,ts in enumerate(t):
        if z<=uppt:
            cse  = DCS(g,m_e,e,ts,1)
            r[z] = np.trapz(sp*cse,e)
        else:
            cse  = DCS(g,m_e,e[k:],ts,1)
            r[z] = np.trapz(sp[k:]*cse,e[k:])
            k    = k + 1
            
    for i in range(len(data_su)):
        num_event[i] = np.trapz(r*res[i],t)
        
    res_tot = ResSu(np.array([[data_su[0,0],data_su[-1,1]]]), t)
    return num_event,np.trapz(r*res_tot[0],t)

def SuperkTotalEventPrediction(dr_dldt,frame):
    year     = frame.year
    theta    = frame.theta
    detector = frame.det_su
    len_m12  = len(dr_dldt)
    res = ResSu(np.array([[frame.data_su[0,0],frame.data_su[-1,1]]]), frame.t_e['B8'][0])
    num_event= np.zeros((len_m12))
    for i in range(len_m12):
        dr_dl        =  detector * np.trapz(dr_dldt[i]['B8'][0]*res[0],frame.t_e['B8'][0],axis=1)
        num_event[i] = np.trapz(dr_dl,theta,axis=0)/year
    return num_event
    
def BorexinoTotalEventPrediction(dr_dldt,t,year,theta):
    #Borexino : number of target per 100 ton :  3.307 \times 10^{31}
    detector  =  24. * 6. * 6. * 0.03307  #number of target per 100 ton per day times 10^{35}
    num_event = 0
    for i in range(len(t)):
        dr_dt     =  (detector/year) * np.trapz(dr_dldt[i],theta,axis=0)
        num_event = num_event + np.trapz(dr_dt,t[i])
    return num_event
    
def SuperkSpectrumEventPrediction(dr_dldt,t,year,theta,detector,b8_un,res):
    num_event = np.zeros((theta.shape[0],b8_un.shape[0]))
    for i in range(b8_un.shape[0]):
        num_event[:,i] = (detector/b8_un[i]) * np.trapz(dr_dldt*res[i],t,axis=1)
    return np.trapz(num_event,theta,axis=0)/year

def AveragedPerdiction(dr_dldt,frame):
    t_e        = frame.t_e
    year       = frame.year
    theta      = frame.theta
    det_su     = frame.det_su
    b8_un      = frame.borom_unoscilated_spectrum
    res        = frame.res
    components = frame.components
    len_m12    = len(dr_dldt)
    pred_bo    = np.zeros((len_m12,3))
    pred_su    = np.zeros((len_m12,b8_un.shape[0]))
    for i in range(len_m12):
        #Borexino
        for k,c in enumerate (components[:-1]):
            pred_bo[i,k] = BorexinoTotalEventPrediction(dr_dldt[i][c],t_e[c],year,theta)
        #SuperKamiokande
        pred_su[i] = SuperkSpectrumEventPrediction(dr_dldt[i]['B8'][0],t_e['B8'][0],year,theta,det_su,b8_un,res)
    return pred_bo,pred_su
    
    
def BorexinoRecoilSpectrum(dr_dldt,t_e,year,theta):
    #Borexino : per 100 ton :  3.307 \times 10^{31}
    detector =  24. * 6. * 6. * 0.03307  #number of target per 100 ton per day times 10^{35}
    dr_dt    = (detector/year) * np.trapz(dr_dldt,theta, axis=0)
    #number of event per day per 100 ton
    cond = t_e>0.02 #Borexino thereshhold
    res  = ResBo(t_e[cond], n_thl = 90, n_thu = 950)
    return np.trapz(res*dr_dt[cond],t_e[cond])
