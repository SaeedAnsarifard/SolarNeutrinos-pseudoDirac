import numpy as np
from scipy.interpolate import interp1d

from framework import SurvivalProbablity,SunEarthDistance,ResBo

def AnnualVariation(frame,mumi,borexino_prediction=False):
    frame.resolution = 0.001
    t    = t_e['Be7'][0]
    cond = t>0.02 #Borexino thereshhold
    res  = ResBo(t[cond])
    dr_dldt  = frame[frame.param['T12'],mumi]]
    dr_dldt  = dr_dldt[i]['Be7'][0]
    
#    enu_be7                 = np.array([frame.e_nu['Be7'][1]])
#    t_be7                   = frame.t_e['Be7'][:,1]
#    resoluton_function      = ResBo(t_be7)
#    ecross_sec              = frame.cs['e']['Be7'][1]
#    mucros_sec              = frame.cs['mu/tau']['Be7'][1]
#
#    l,a,theta,h             = SunEarthDistance(0.001)
#    sorted_l,sorted_theta,sorted_day = ThetatoDay(theta,l,h,a)
#
#    be7_pee,be7_pes  = SurvivalProbablity(frame.phi['Be7'],enu_be7,frame.n_e,frame.f_c,frame.hbarc,frame.param,sorted_l)
    
    rtbe7 = np.zeros(be7_pee.shape[0])
    for i in range(be7_pee.shape[0]):
        rtbe7[i] = np.trapz(resoluton_function*(ecross_sec*be7_pee[i] + mucros_sec*(1-be7_pes[i]-be7_pee[i])),t_be7)
        
    r_ave = np.trapz(rtbe7[np.argsort(sorted_theta)],np.sort(sorted_theta))  
    r_ave = (frame.time/frame.year) * frame.det_bo * frame.norm['Be7'] * (a**2/h) * r_ave
    r_day = frame.time * frame.det_bo * frame.norm['Be7'] * (a/sorted_l)**2 * rtbe7
   
    if borexino_prediction:
        data_index  = DataIndex()
        r_dataindex = np.zeros((2,116))

        for i in range(116):
            r_dataindex[0,i] = data_index[0,i]
            collect          = []
            for j in range(1,31):
                collect.append(r_day[int(data_index[j,i]-1)])
            r_dataindex[1,i] = np.mean(collect)
        r_dataindex[1,:] = r_dataindex[1,:] - r_ave
        return r_dataindex
    else:
        return np.array([sorted_day,r_day - r_ave])

def ThetatoDay(theta,l,h,a):
    e      = (1.521e11 - 1.471e11)/(1.521e11 + 1.471e11)
    day    = np.zeros(theta.shape[0])
    day[0] = 4. #Earth is at perihelion on January 4
    for i in range(1,theta.shape[0]):
        day_counter = np.trapz(l[:i]**2,theta[:i])/(h*60.*60.*24.) + 4.  
        if day_counter <= 365.25 :
            day[i] =  day_counter
        else:
            day[i] =  day_counter - 365.25
    
    f            = interp1d(day, theta)
    sorted_day   = np.arange(1,366)
    sorted_theta = f(sorted_day)
    cos          = np.cos(sorted_theta)
    sorted_l     = a*(1-e**2)/(1+e*cos)
    return sorted_l,sorted_theta,sorted_day
    
def DataIndex():
    one_binning    = np.arange(1,366)
    thirty_binning = [[]for i in range(119)]
    for i in range(341,3570+341):   #341 and 3570+341 are corresponding to 11 dec 2011 to 3 Oct 2021
        thirty_binning[(i-341)//30].append(one_binning[np.mod(i,365)])

    data_index = np.zeros((31,116))
    k = 0
    for i in range(116):
        if i==54:
            k=k+2
        if i==90:
            k=k+1

        data_index[0,i]  = 15 + 30 * k
        data_index[1:,i] = thirty_binning[k]
        k = k + 1
    return data_index
