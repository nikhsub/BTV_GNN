#!/bin/bash

for i in {1..16}
do
	 eos root://cmseos.fnal.gov mv /store/group/lpcljm/nvenkata/BTVH/toprocfiles/test/output_11_chunk${i}.root /store/group/lpcljm/nvenkata/BTVH/toprocfiles/test/excess/output_11_chunk${i}.root	
 done
