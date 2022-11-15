import argparse
from chi2finder import Chi2Finder
from framework import FrameWork

parser = argparse.ArgumentParser(description='Join CHi2 Analsys for constraning the psudo-Diac scheme by Super-Kamiokande and Borexino Data',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--su_nbins', default=13, type=int, metavar='N',help='number of bins for Super-Kamiokande experiment')
parser.add_argument('--m12', default=7.54e-10, type=float, metavar='M',help='active neutrino mass splitting')
parser.add_argument('--t12_var', default=5., type=float, metavar='M',help='initial variance of theta_{12}')
parser.add_argument('--iteration', default=5, type=int, metavar='N',help='number of iteration over theta_{12} sample')
parser.add_argument('--t12_sample', default=50, type=int, metavar='N',help='number of theta_{12} sample at each mumi step')
parser.add_argument('--mumi', type=str, choices=['mum1', 'mum2'], default='mum2',help='Majorana Dirac mass splitting')
parser.add_argument('--m12_nuisance', action=argparse.BooleanOptionalAction,default=True, help='including M12 as a nuisance from KamLand')
parser.add_argument('--low_resolution', action=argparse.BooleanOptionalAction,default=True, help='less number of mumi but running faster')


def main():
    args = parser.parse_args()
    frame = FrameWork(args.su_nbins,args.mumi,args.m12,args.m12_nuisance)
    chi2  = Chi2Finder(args.t12_var, args.iteration, args.t12_sample, args.low_resolution)
    chi2.run(frame) 


if __name__ == '__main__':
    main()
