import argparse
import numpy as np

from chi2finder import Chi2Finder
from framework import FrameWork

parser = argparse.ArgumentParser(description='Join CHi2 Analsys for constraning the psudo-Diac scheme by Super-Kamiokande and Borexino Data',formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--su_nbins', default=13, type=int, metavar='N',help='number of bins for Super-Kamiokande experiment')
parser.add_argument('--m12', default=7.54e-5, type=float, metavar='M',help='active neutrino mass splitting')
parser.add_argument('--t12', default=33.4, type=float, metavar='M',help='active neutrino theta_{12}')
parser.add_argument('--t12_var', default=5., type=float, metavar='M',help='initial variance of theta_{12}')
parser.add_argument('--pow_low', default=-13, type=float,choices=[-13,-12,-11], metavar='M',help='the initial power of mumi')
parser.add_argument('--pow_high', default=-10.3, type=float,choices=[-12,-11,-10.3], metavar='M',help='final power of mumi')
parser.add_argument('--iteration', default=5, type=int, metavar='N',help='number of iteration over theta_{12} sample')
parser.add_argument('--t12_sample', default=50, type=int, metavar='N',help='number of theta_{12} sample at each mumi step')
parser.add_argument('--mumi', type=str, choices=['mum1', 'mum2'], default='mum2',help='Majorana Dirac mass splitting')
parser.add_argument('--m12_nuisance', action=argparse.BooleanOptionalAction,default=True, help='including M12 as a nuisance from KamLand')
parser.add_argument('--low_resolution', action=argparse.BooleanOptionalAction,default=True, help='less number of mumi but running faster')


def main():
    args  = parser.parse_args()
    if args.pow_low >= args.pow_high:
          raise Exception('Error : pow_high <= pow_low')
    frame = FrameWork(args.su_nbins,args.mumi,args.t12,args.m12,args.m12_nuisance)
    chi2  = Chi2Finder(args.t12_var, args.iteration, args.t12_sample, args.pow_low, args.pow_high, args.low_resolution)
    chain = chi2.run(frame) 
    np.savetxt('./Result/chain_'+str(args.mumi)+'_'+str(np.abs(args.pow_low))+'_'+str(np.abs(args.pow_high))+'.txt',chain.T,fmt='   '.join(['%i']+['%1.1f']+['%1.1f']+['%1.3e']), header='weight  -log(liklihood) T12  muj ')


if __name__ == '__main__':
    main()
