#!/bin/csh

nmrPipe -in test.fid \
| nmrPipe -fn EM -lb 0.3 \
| nmrPipe -fn ZF -auto \
| nmrPipe -fn FT -auto \
| nmrPipe -fn PS -p0 25.2 -p1 -56.4 -di -verb \
  -ov -out spectrum.ft

nmrPipe -in test.fid \
| nmrPipe -fn FT \
| nmrPipe -fn PS -p0 25.2 -p1 -56.4 -verb \
| nmrPipe -fn FT -inv \
  -ov -out fid_phased.fid
