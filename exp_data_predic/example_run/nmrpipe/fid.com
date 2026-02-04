#!/bin/csh

bruk2pipe -verb -in ./ser \
  -bad 0.0 -ext -aswap -DMX -decim 2380 -dspfvs 21 -grpdly 76  \
  -xN             40960  -yN                 8  \
  -xT             20480  -yT                 8  \
  -xMODE            DQD  -yMODE           Real  \
  -xSW         8403.361  -ySW         6009.615  \
  -xOBS         600.253  -yOBS         600.253  \
  -xCAR           4.811  -yCAR           4.797  \
  -xLAB             1Hx  -yLAB             1Hy  \
  -ndim               2  -aq2D       Magnitude  \
| nmrPipe -fn MULT -c 1.95312e+00 \
  -out ./test.fid -ov

