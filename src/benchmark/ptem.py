# script to compute the Program Time-Error Metric
# takes as input the reference and hypothesis files in stm format as well as the collar in milliseconds

import sys, getopt
import statistics
#import time
import matplotlib.pyplot as plt


def main(argv):
    inputfile = ''
    outputfile = ''
    name='file'
    punt=0
    collar=0
    try:
        opts, args = getopt.getopt(argv,"h:r:c:")
    except getopt.GetoptError:
        print('ptem.py -r <stm reference file> -h <stm hyphotesis file> -c <collar in ms| def: 0ms>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h" ):
            hypfile = arg
        elif opt in ("-r"):
            reffile = arg
        elif opt in ("-c"):
            collar = float(arg)/1000.

    print("computing PTEM... ")

    refdata=[]
    hypdata=[]

    with open(reffile,"r") as fr:
        for lr in fr:
            r = lr.strip().split("<,,>")      
            refdata.append([r[0].split()[-2],r[0].split()[-1],r[1].strip()])
    with open(hypfile,"r") as fh:
        for lh in fh:
            h = lh.strip().split("<,,>")
            hypdata.append([h[0].split()[-2],h[0].split()[-1],h[1].strip()])

    if len(hypdata)!=len(refdata):
        print("ERROR, files of diferent lengths",len(hypdata),len(refdata))
    else:
        print("Number of subtitles: ",len(hypdata))
    Tini=[]
    Tfin=[]
    TE=[]
    for ref,hyp in zip(refdata,hypdata):
        ini=abs(float(ref[0])-float(hyp[0]))
        if ini<collar:
            ini=0.0
        fin=abs(float(ref[1])-float(hyp[1]))
        if fin<collar:
            fin=0.0
        Tini.append(ini)
        Tfin.append(fin)
        TE.append(ini+fin)
    print("Programa Time-Error Metric (PTEM)")
    print("MEDIAN start time \t %5.2f seconds"%statistics.median(Tini))
    print("MEDIAN end time \t %5.2f seconds"%statistics.median(Tfin))
    print("MEDIAN program \t\t %5.2f seconds"%statistics.median(TE))
    print("MEAN program \t\t %5.2f seconds"%statistics.mean(TE))

    
    #plt.hist(TE,bins=50)
    #plt.show()

if __name__ == "__main__":
   main(sys.argv[1:])
