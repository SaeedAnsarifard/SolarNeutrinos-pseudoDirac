import numpy as np

class FrameWork(object):
    #Su_nbin is the number of Super-Kamiokande bins you need process. Max is 23
    #mumi is \Delta m_{i}^2 and has two options 'mum2' or 'mum1'
    #M12 is the mean value of \Delta m_{12}^2 
    #If M12_Nusiance='True' then M12 included as nusiance paramtere else it fixed to M12 value
    def __init__(self,Su_nbin=23, mumi = 'mum2', m12=7.54e-5, m12_nusiance='True'):
        #Fermi constant :  1.166 \times 10^{-11}/Mev^2
        self.GF    = 1.166
        #Electron mass : 0.511 Mev
        self.m_e    = 0.511
        #h_{bar}c : 1.97 10^{-11} Mev.cm
        self.hbarc = 1.97
        #proton mass :  1.67 \times 10^{-27} kg
        self.m_p    = 1.67 
        #Nutrino Flux normalization :    Arxiv : 1611.09867 
        self.norm  = {'pp' : 5.98,
                      'Be7': 4.93e-1,
                      'pep': 1.44e-2, 
                      'B8' : 5.46e-4}   #\times 10^{10}
        #Neutrino production point weight function : http://www.sns.ias.edu/~jnb/
        load_phi   = np.loadtxt('./Solar_Standard_Model/bs2005agsopflux1.txt', unpack = True)
        self.phi   = {'pp' : load_phi[5,:],
                      'Be7': load_phi[10,:],
                      'pep': load_phi[11,:],
                      'B8' : load_phi[6,:]}
        #Electron density inside sun 10^{23}/cm^{3}
        self.n_e  = 6*10**load_phi[2,:]
        #Neutrino energy spectrum : http://www.sns.ias.edu/~jnb/
        spectrumB8      = np.loadtxt('./Data/8B_Spectrum.txt')
        spectrumB8[:,1] = spectrumB8[:,1]/np.trapz(spectrumB8[:,1],spectrumB8[:,0])
        spectrumpp      = np.loadtxt('ppenergytab1.txt')
        self.spec       = {'pp' : spectrumpp[:,1],
                           'Be7': np.array([1,1]),
                           'pep': np.array([1])  ,
                           'B8' : spectrumB8[:,1]}
        #Neutrino energy in Mev
        self.E          = {'pp' : spectrumpp[:,0]        ,
                           'Be7': np.array([0.384,0.862]),
                           'pep': np.array([1.44])       ,
                           'B8' : spectrumB8[:,0]        }
        #electron recoil energy in Mev
        self.T   = {'pp' : self.E['pp'] /(1 + self.me /(2 * self.E['pp'])),
                    'Be7': np.linspace(0.05,self.E['Be7']/(1+self.me/(2*self.E['Be7'])),100),
                    'pep': np.linspace(0.05,self.E['pep'][0]/(1+self.me/(2*self.E['pep'][0])),100),
                    'B8' : self.E['B8'] /(1 + self.me /(2 * self.E['B8']))}
        #Borexino Data event (count per day per 100 ton) : https://doi.org/10.1038/s41586-018-0624-y 
        #Eur. Phys. J. C 80, 1091 (2020)
        self.data_Bo = {'pp' : {'R' : 134.,'e' : 20.},
                        'Be7': {'R' : 48.3,'e' : 1.8},
                        'pep': {'R' : 2.43,'e' : 0.58}}
        #Super-K Data event (count per year per  kilo ton) :
        self.data_Su  = np.loadtxt('./Data/B8_Data_2020.txt')[:Su_nbin,:]
        #detector normalization 
        #Borexino : per 100 ton :  3.307 \times 10^{31} 
        #Super-K  : per Kton    :  (10/18) \times 10^{6}/m_p
        self.det_Bo = 0.03307              
        self.det_Su = (10/18) * 1/self.mp
        #event per day  : 24 \times 60 \times 60 
        self.time   = 24.*6.*6.     
        self.L,self.a,self.theta,self.H   = Sun_Earth_distance()
        self.year  = 60*60*24*365.25
		#electron recoil cross section
        G        = self.hbarc * self.GF
        self.CS  = {'e' : {'pp' : [dCS(G,self.me,self.E['pp'][i:],t,1) for i,t in enumerate (self.T['pp'])], 
                           'Be7': [dCS(G,self.me,self.E['Be7'][i],self.T['Be7'][:,i],1) for i in range(2)],  
                           'pep': dCS(G,self.me,self.E['pep'][0],self.T['pep'],1),
                           'B8' : [dCS(G,self.me,self.E['B8'][i:],t,1) for i,t in enumerate (self.T['B8'])]}, 
                    'mu/tau' : {'pp' : [dCS(G,self.me,self.E['pp'][i:],t,-1) for i,t in enumerate (self.T['pp'])], 
                                'Be7': [dCS(G,self.me,self.E['Be7'][i],self.T['Be7'][:,i],-1) for i in range(2)],  
                                'pep': dCS(G,self.me,self.E['pep'][0],self.T['pep'],-1),
                                'B8' : [dCS(G,self.me,self.E['B8'][i:],t,-1) for i,t in enumerate (self.T['B8'])]}}
        
	    #Super-k detector response function   
        self.res = Res_Su(self.Data_Su,self.T['B8'])
        
        self.t13  = 8.57
        self.mum3 = 0.
        self.mumi = mumi
        
        shape = np.loadtxt('./Correlated_Errors/Nutrino_shape_Systematic_Uncertainties.txt')[:Su_nbin,:]        
        scale = np.loadtxt('./Correlated_Errors/Energy_Scale_Systematic_Uncertainties.txt')[:Su_nbin,:]
        resol = np.loadtxt('./Correlated_Errors/Energy_Resolution_Systematic_Uncertainties.txt')[:Su_nbin,:]

        self.delta = np.random.normal(0,1,(500,3))

        self.f     = ((1/(1+self.delta[:,0,np.newaxis]*shape[:,1]/100))*
                      (1/(1+self.delta[:,1,np.newaxis]*scale[:,1]/100))*
                      (1/(1+self.delta[:,2,np.newaxis]*resol[:,1]/100)))
		
		
	    #Base on KamLAND
        self.m12_bar  = 7.54e-5
        self.sig_m12  = 5.0e-6
        if m12_nusiance:
            self.m12 = np.linspace(self.m12_bar-(2*self.sig_m12),self.m12_bar+(2*self.sig_m12),7)
            
        else:
            self.m12 = np.array([m12])
			
        self.pred_Bo = np.zeros((self.m12.shape[0],3))
        self.pred_Su = np.zeros((self.m12.shape[0],self.data_Su.shape[0]))
         
    def __getitem__(self,param_ubdate):
        param  = {'T12' : param_ubdate[0],
                  'T13' : self.t13, 
                  'mum1': 0. ,
                  'mum2': 0. ,
                  'mum3': self.mum3,
                  'M12' : 0. }
        param[self.mumi] = param_ubdate[1]
        
        for i in range(self.M12.shape[0]):
            param['M12'] = self.m12[i]
            
            pep_Pee,pep_Pes=survival_probablity(self.phi['pep'],self.E['pep'],self.n_e,self.GF,self.hbarc,param,self.L)
            Be7_Pee,Be7_Pes=survival_probablity(self.phi['Be7'],self.E['Be7'],self.n_e,self.GF,self.hbarc,param,self.L)
            pp_Pee,pp_Pes  = survival_probablity(self.phi['pp'],self.E['pp'],self.n_e,self.GF,self.hbarc,param,self.L)
            B8_Pee,B8_Pes  = survival_probablity(self.phi['B8'],self.E['B8'],self.n_e,self.GF,self.hbarc,param,self.L)


            Pee = {'pp' : pp_Pee, 'Be7' : Be7_Pee, 'pep' : pep_Pee, 'B8' : B8_Pee} 
            Pes = {'pp' : pp_Pes, 'Be7' : Be7_Pes, 'pep' : pep_Pes, 'B8' : B8_Pes} 
            Rpep,RBe7,Rpp,RB8 = event_rate_maker(Pee,Pes,self.CS,self.E,self.T,self.L,
                                                 self.spec,self.theta,self.data_Su,self.res)



            self.pred_Bo[i]=(self.time/self.year)*self.det_Bo* (self.a**2/self.H) * np.array([self.norm['pp']*Rpp,
                                                                                              self.norm['Be7']*RBe7,
                                                                                              self.norm['pep']*Rpep])
            self.pred_Su[i]= 365.*(self.time/self.year)*self.det_Su*(self.a**2/self.H)*(self.norm['B8']*
                                                                                    RB8/self.data_Su[:,-1])
            
        return Chi2(self.pred_Bo,self.pred_Su,self.data_Bo,self.data_Su,self.f,self.delta,self.m12,self.m12_bar,self.sig_m12)

def Sun_Earth_distance(resolution=0.08):
    a     = (1.521e11 + 1.471e11)/2  #L = 1.5e11 meter 
    e     = (1.521e11 - 1.471e11)/(1.521e11 + 1.471e11)

    theta = np.arange(0,2*np.pi,resolution)
    cos   = np.cos(theta)
    L     = a*(1-e**2)/(1+e*cos)    
    H     = np.trapz(L**2,theta)/(60*60*24*365.25)
    return L,a,theta,H
    
    
def dCS(G,me,E,T,i):
    #dsigma/dT as function of T and E (electron recoil and neutrino energy)

    #weak mixing angle = 0.22342 : https://pdg.lbl.gov/2019/reviews/rpp2019-rev-standard-model.pdf
    sw  = 0.22342
    y   = T/E
    A1  = (2 * sw + i)**2
    A2  =  4 * sw**2 * (1 - y)**2
    A3  =  2 * sw * (2 * sw + i) * me * y/E
    return  G**2 * (me/(2*np.pi)) * (A1+A2-A3) * 10 #\times 10^{-45} in cm^2


def Res_Su(D,T):
    R   = np.zeros((D.shape[0],T.shape[0]))
    for j in range (D.shape[0]):
        E = np.linspace(D[j,0],D[j,1])
        for i,t in enumerate(T):
            sig  = (-0.084+0.349*np.sqrt(t)+0.04*t)
            A    = (1/(np.sqrt(2*np.pi)*sig))*np.exp(-0.5*(t-E)**2/sig**2)
            R[j,i] = np.trapz(A,E)
    return R

def survival_probablity(phi,Es,Ne,GF,hbarc,param,Ls): 
    PeL   = np.zeros((Ls.shape[0],Es.shape[0]))
    PsL   = np.zeros((Ls.shape[0],Es.shape[0]))
    
    for j,L in enumerate(Ls):
        Util= np.ones((Ne.shape[0],Es.shape[0]))
        ne  = np.reshape(Ne ,(Ne.shape[0],1))*Util
        E   = np.reshape(Es ,(1,Es.shape[0]))*Util

        VE  = 2*np.sqrt(2)*GF*ne*hbarc**3*1e-9 * E

        Den = np.sqrt((param['M12']*np.cos(2*(np.pi/180)*param['T12'])-VE)**2 + 
                      (param['M12']*np.sin(2*(np.pi/180)*param['T12']))**2)
        Nom = param['M12']*np.cos(2*(np.pi/180)*param['T12']) - VE

        TM  = 0.5*np.arccos(Nom/Den)

        sin = np.sin((np.pi/180)*param['T12'])**2 * np.cos((np.pi/180)*param['T13'])**4
        cos = np.cos((np.pi/180)*param['T12'])**2 * np.cos((np.pi/180)*param['T13'])**4

        Ae1 = cos * np.cos(TM)**2  * np.cos(10*param['mum1']*L/(hbarc*2*E))**2
        Ae2 = sin * np.sin(TM)**2  * np.cos(10*param['mum2']*L/(hbarc*2*E))**2
        Ae3 = np.sin((np.pi/180)*param['T13'])**4

        pee = Ae1 + Ae2 + Ae3
        PeL[j]  = np.sum(np.reshape(phi,(Ne.shape[0],1))*pee,axis=0)

        As1 = np.cos((np.pi/180)*param['T13'])**2 * np.cos(TM)**2  * np.sin(10*param['mum1']*L/(hbarc*2*E))**2
        As2 = np.cos((np.pi/180)*param['T13'])**2 * np.sin(TM)**2  * np.sin(10*param['mum2']*L/(hbarc*2*E))**2
        As3 = np.sin((np.pi/180)*param['T13'])**2 * np.sin(10*param['mum3']*L/(hbarc*2*E))**2

        pes = As1 + As2 + As3
        PsL[j]  = np.sum(np.reshape(phi,(Ne.shape[0],1))*pes,axis=0)

    return PeL, PsL

def event_rate_maker(Pee,Pes,CS,E,T,L,spec,theta,Data_Su,Res):
    Rpep_L = np.zeros(L.shape[0])
    RBe7_L = np.zeros((L.shape[0],2))
    Rpp_T  = np.zeros((L.shape[0],E['pp'].shape[0]))
    Rpp_L  = np.zeros(L.shape[0])
    RB8_T  = np.zeros((L.shape[0],T['B8'].shape[0]))
    RB8_L  = np.zeros((L.shape[0],Data_Su.shape[0]))
    RB8    = np.zeros((Data_Su.shape[0]))
    
    Rpep_L = np.trapz((CS['e']['pep']*Pee['pep'][:,0,np.newaxis] +
                       CS['mu/tau']['pep']*(1-Pee['pep'][:,0,np.newaxis]-Pes['pep'][:,0,np.newaxis])),T['pep'],axis=1)
    
    Rpep   =  np.trapz(Rpep_L,theta)
    
    for i in range(2):
        RBe7_L[:,i] = np.trapz((CS['e']['Be7'][i]*Pee['Be7'][:,i,np.newaxis] + 
                                CS['mu/tau']['Be7'][i]*(1-Pee['Be7'][:,i,np.newaxis]-Pes['Be7'][:,i,np.newaxis])),
                               T['Be7'][:,i],axis=1)
    RBe7 =  (0.1*np.trapz(RBe7_L[:,0],theta) + np.trapz(RBe7_L[:,1],theta))

    for i,t in enumerate(T['pp']):
        Rpp_T[:,i] = np.trapz(spec['pp'][np.newaxis,i:]*
                              (CS['e']['pp'][i][np.newaxis,:]*Pee['pp'][:,i:] + 
                               CS['mu/tau']['pp'][i][np.newaxis,:]*(1-Pee['pp'][:,i:]-Pes['pp'][:,i:])),
                              E['pp'][i:],axis=1)
    Rpp_L = np.trapz(Rpp_T,T['pp'],axis=1)  
    Rpp   =  np.trapz(Rpp_L,theta)

    for i,t in enumerate(T['B8']):
        CS1 = CS['e']['B8'][i][np.newaxis,:]
        CS2 = CS['mu/tau']['B8'][i][np.newaxis,:]
        RB8_T[:,i] = np.trapz(spec['B8'][i:]*(CS1*Pee['B8'][:,i:] + 
                                              CS2*(1-Pee['B8'][:,i:]-Pes['B8'][:,i:])),E['B8'][i:])
    for k in range(Data_Su.shape[0]):
        RB8_L[:,k] = np.trapz(Res[k]*RB8_T,T['B8'],axis=1)

    RB8  =  np.trapz(RB8_L,theta,axis=0)
    return Rpep,RBe7,Rpp,RB8


def Chi2(pred_bo,pred_su,data_bo,data_su,f,delta,M12,M12_bar,sig_M12):
    #Flux normalization uncertainties taking from solar standrad model predection  
    sig_norm_bo = np.array([0.01,0.06,0.01])
    sig_norm_su = 0.12
    
    D_Bo = np.array([data_bo['pp']['R'],data_bo['Be7']['R'],data_bo['pep']['R']])
    E_Bo = np.array([data_bo['pp']['e'],data_bo['Be7']['e'],data_bo['pep']['e']])
    D_Su = data_su[:,2]
    E_Su = data_su[:,3]
    
    chi = np.zeros(M12.shape)
    for i in range(M12.shape[0]):
        min_norm_bo = (E_Bo**2 + (D_Bo*pred_bo[i]*sig_norm_bo**2))/(E_Bo**2 + (pred_bo[i]**2*sig_norm_bo**2))

        sig_frac = (sig_norm_su/E_Su)**2
        nominator= D_Su*pred_su[i]*sig_frac
        denominat= pred_su[i]**2*sig_frac

        min_norm_su = (np.sum(f*nominator[np.newaxis,:],axis=1)+1)/(np.sum(f**2*denominat[np.newaxis,:],axis=1)+1)
        
        chi_Bo      = np.sum((D_Bo - min_norm_bo*pred_bo[i])**2/E_Bo**2 + (min_norm_bo - 1)**2/sig_norm_bo**2)
        
        
        A = np.ones((f.shape[0],1))*D_Su[np.newaxis,:] - f * (min_norm_su[:,np.newaxis] * pred_su[i,np.newaxis,:])
        B = np.ones((f.shape[0],1))*E_Su[np.newaxis,:]
        
        
        chi_Su = np.sum((A/B)**2,axis=1) + np.sum(delta**2,axis=1) 
        
        chi_Su = chi_Su + ((min_norm_su - 1)/sig_norm_su)**2

        chi[i] = np.min(chi_Su) + chi_Bo + ((M12[i] - M12_bar)/sig_M12)**2 
        
    return np.min(chi)
