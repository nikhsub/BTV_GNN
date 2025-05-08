#!/bin/bash

for i in {1..16}
do
	 eos root://cmseos.fnal.gov mv /store/group/lpcljm/nvenkata/BTVH/toprocfiles/output_1_chunk${i}.root /store/group/lpcljm/nvenkata/BTVH/toprocfiles/processed/output_1_chunk${i}.root	
 done
