import argparse

from chi2finder import Chi2Finder

from framework import FrameWork

parser = argparse.ArgumentParser(description='Join CHi2 Analsys for constraning the psudo-Diac scheme by Super-Kamiokande and Borexino Data',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--Su_nbins', default=13, type=int, metavar='N',help='number of bins for Super-Kamiokande experiment')
parser.add_argument('--M12', default=7.54e-10, type=float, metavar='M',help='active neutrino mass spliting')
parser.add_argument('--mumi', type=str, choices=['mum1', 'mum2'], default='mum2',help='Majorana Dirac mass spliting')
parser.add_argument('--M12_Nusiance', action='store_true',default=True, help='including M12 as a nusiance from KamLand')


def main():
    args = parser.parse_args()
    frame = FrameWork(args.Su_nbins,args.mumi,args.M12,args.M12_Nusiance)
    chi2  = Chi2Finder()
    chi2.run(frame) 


if __name__ == '__main__':
    main()
