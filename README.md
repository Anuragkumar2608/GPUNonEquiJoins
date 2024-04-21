The goal of this project was to use GPUs to compute non-equi joins on integer attributes. Specifically, given a pair of tables R(ridX, IntA) and S(ridY, IntB), where rid are 32-bit integer row identifiers, and Int are integer attributes, compute the matching pairs (R.ridX, S.ridY ) where (R.IntA < S.IntB).

generate_csv.cpp - generates the two CSV files that will be used as the two tables to get input.

rangejoin.cu - CUDA program to compute the join of the two tables. 
The test system has 4 gigabytes of GPU memory, so if the size of the join result is less than 2GB, then entire join process happens parrallely.
Otherwise, if the size of the result is greater than 2GBs, then we perform the join batch wise instead of computing the entire thing at the same time.