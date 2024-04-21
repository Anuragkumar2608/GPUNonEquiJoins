#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <numeric>
#include <chrono>

typedef long long int lld;

#define R_SIZE 1000000
#define S_SIZE 1000000
#define BLOCK_SIZE 1024

struct Record
{
    int rid;
    int value;
};

struct Pair {
    int first;
    int second;
};

struct compareRecords
{
    __host__ __device__ bool operator()(const Record &a, const Record &b)
    {
        // Compare based on the value field
        return a.value < b.value;
    }
};

//Function to convert each array element to store the number of results for the corresponding array element
//in R to compute exclusive scan to get the size of the results for each element.
__global__ void convert(int* incpy){
     int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("%d \n", tid);

    // Calculate the index for the result array
    int index = tid;

    // Ensure the thread is within the valid range
    if(index < R_SIZE) {
        incpy[index] = R_SIZE - incpy[index];
    }else{
        return;
    }
}

// Computes pairs with each element of R by spawning a thread for each element of R.
__global__ void makeAllPairs(const Record *S, const Record *R, int* indices, int* incpy, Pair* result){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("%d \n", tid);

    // Calculate the index for the result array
    //int index = tid;

    // Ensure the thread is within the valid range
    if(index < R_SIZE){
        int counter = 0;
        for(int i=indices[index]; i<S_SIZE; i++){
            result[incpy[index] + counter].first = R[index].rid;
            result[incpy[index] + counter++].second = S[i].rid;
        }
    }else{
        return;
    }
}

// Computes pairs in batches, threads are equal to the number of pairs present in each batch.
__global__ void makePairsWithXKernel(const Record *S,const Record *R, Pair* result, int n, int ind) {
    int x = R[ind].rid;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("%d \n", tid);

    // Calculate the index for the result array
    //int index = tid;

    // Ensure the thread is within the valid range
    if (index + n < S_SIZE) {
        // Store the pair in the result array
        result[index].first = x;
        result[index].second = S[index + n].rid;
    }else{
        return;
    }
}

// Reads the CSV file to get the corresponding records.
std::vector<Record> readCSV(const std::string &filename)
{
    std::vector<Record> records;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line))
    {
        Record record;
        if (sscanf(line.c_str(), "%d,%d", &record.rid, &record.value) == 2)
        {
            records.push_back(record);
        }
    }

    file.close();
    return records;
}

// Performs binary search for each element in R to compute the index after
// which each element of S is greater than current element of R.
__global__ void parallelBinarySearch(const Record *R, const Record *S, int* results, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        int left = 0;
        int right = size-1;
        int target = R[tid].value;
        results[tid] = S_SIZE;
        while (left <= right) {
            int mid = (left + right) / 2;
            int currval = S[mid].value;
            if (currval > target)
            {
                results[tid] = mid;
                right = mid - 1;
            }
            else
            {
                left = mid + 1;
            }
        }
        //if(right<0) results[tid] = -1;
        //if(left>right) results[tid] = right;
        //results[tid] = binarySearch(S, R[tid].value, 0, size - 1);
    }else{
        return;
    }
}



int main() {
    //std::cout<<"Starting main"<<std::endl;
    // Start measuring time
    //auto begin = std::chrono::high_resolution_clock::now();

    std::vector<Record> a = readCSV("R_table.csv");
    std::vector<Record> b = readCSV("S_table.csv");

    std::cout<<"Done reading files"<<std::endl;
    auto begin = std::chrono::high_resolution_clock::now();

    thrust::device_vector<Record> R = a;
    thrust::device_vector<Record> S = b;
    std::cout<<"Done creating thrust vectors"<<std::endl;

    int *indices;
    cudaMalloc((void **)&indices, R.size() * sizeof(int));
    std::cout<<"Done cudaMalloc for indices"<<std::endl;

    auto afterMemCpy = std::chrono::high_resolution_clock::now();
    auto timeAfterMemCpy = std::chrono::duration_cast<std::chrono::nanoseconds>(afterMemCpy - begin);
    printf("Time measured after MemCpy: %.3f ms.\n", timeAfterMemCpy.count() * 1e-6);
    
    std::cout<<"Just before sorting"<<std::endl;
    thrust::sort(S.begin(), S.end(), compareRecords());
    std::cout<<"Done sorting"<<std::endl;
    auto afterSort = std::chrono::high_resolution_clock::now();
    auto timeAfterSort = std::chrono::duration_cast<std::chrono::nanoseconds>(afterSort - begin);
    printf("Time measured after Sort: %.3f ms.\n", timeAfterSort.count() * 1e-6);
    
    int numBlocks = (R.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Record *R_ptr = thrust::raw_pointer_cast(R.data());
    Record *S_ptr = thrust::raw_pointer_cast(S.data());
    
    std::cout<<"Started Binary Search"<<std::endl;
    parallelBinarySearch<<<numBlocks, BLOCK_SIZE>>>(R_ptr, S_ptr, indices, R.size());
    //std::cout<<"Reached after binary search\n";
    //lld result = thrust::reduce(thrust::device, indices, indices + R_SIZE);
    
    int* hostRes = (int*)malloc(sizeof(int)*R.size());
    auto afterMalloc = std::chrono::high_resolution_clock::now();
    auto timeAfterMalloc = std::chrono::duration_cast<std::chrono::nanoseconds>(afterMalloc - begin);
    printf("Time measured after Malloc: %.3f ms.\n", timeAfterMalloc.count() * 1e-6);
    
    cudaMemcpy(hostRes, indices, R.size() * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout<<"Reached after memcpy\n";
    lld res = 0;
    for(int i=0; i<R_SIZE; i++){
          res += S_SIZE-hostRes[i];
    }
    std::cout<<"The result contains: "<<res<<" tuples\n";



    if(res > (1<<29)){
        std::cout<<"Inside if."<<"\n";
        //Record *Rcpy = (Record*)malloc(R.size() * sizeof(Record));
        //cudaMemcpy(Rcpy, R.data().get(), R.size() * sizeof(Record), cudaMemcpyDeviceToHost);
        /*for(int i=0; i<R.size(); i++){
            std::cout<<"R[i].rid: "<<Rcpy[i].rid<<"   R[i].value: "<<Rcpy[i].value<<"\n";
        }*/

        //int** array2D = (int**)malloc(rows * sizeof(int*));
        
        Pair **result = (Pair**)malloc(R.size() * sizeof(Pair*));
        std::cout<<"Reached after pair double pointer declaration"<<std::endl;
        // Loop over values of x
        //std::ofstream myfile;
        //myfile.open ("result.csv");
        
        for (int x = 0; x <= R.size(); ++x) {
            
            // Calculate the size of the result vector
            int numPairs = S_SIZE-hostRes[x];

            //If the number of pairs in the iteration is zero, continue
            if(numPairs == 0)continue;

            // Allocate GPU memory for result vector
            Pair* d_result;
            cudaMalloc((void**)&d_result, numPairs * sizeof(Pair));
            //std::cout<<"Reached after cuda malloc of d_result"<<std::endl;

            result[x] = (Pair*)malloc(numPairs * sizeof(Pair));
            // Launch kernel
            //std::cout<<"result[x] address: "<<result[x];
            //std::cout<<"Reached after declaring result[x]"<<std::endl;
            
            int numBlocks = (numPairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
            //std::cout<<"hostRes[x]: "<<hostRes[x]<<" R_ptr[x].rid: "<<Rcpy[x].rid<<std::endl;
            makePairsWithXKernel<<<numBlocks, BLOCK_SIZE>>>(S_ptr, R_ptr, d_result, hostRes[x], x);
            cudaDeviceSynchronize();
            
            cudaError_t cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess) {
                fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
            }


            //std::cout<<"Reached after kernel"<<std::endl;

            // Copy result from GPU to host
            cudaMemcpy(&result[x][0], d_result, numPairs * sizeof(Pair), cudaMemcpyDeviceToHost);

            //std::cout<<"Reached after cuda memcpy"<<std::endl;

            cudaDeviceSynchronize();

            cudaFree(d_result);

            // Print result
            /*std::cout << "Pairs for x = " << x << ": ";
            for (int i = 0 ; i < S_SIZE-hostRes[x]; ++i) {
                 myfile << result[x][i].first << "," << result[x][i].second << "\n";
            }*/
            free(result[x]);

        }

        free(result);

        /*for(int i=0; i<R.size(); i++){
            for(int j=0; j<S_SIZE-1-hostRes[i]; j++){
                std::cout<<"rkey: "<<result[i][j].first<<" skey: "<<result[i][j].second<<"\n";
            }
        }*/

        // std::cout<<"Reached Here";
    }else{
        std::cout<<"Inside else.\n";
        int *incpy;

        cudaMalloc((void**)&incpy, R.size() * sizeof(int));
        cudaMemcpy(incpy, indices, R.size() * sizeof(int), cudaMemcpyDeviceToDevice);
        
        int numBlocks = (R.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        convert<<<numBlocks, BLOCK_SIZE>>>(incpy);
        thrust::device_ptr<int> thrustPtr(incpy);
        thrust::exclusive_scan(thrustPtr, thrustPtr + R.size(), thrustPtr);
        
        int *hcpy = (int*)malloc(R.size() * sizeof(int));
        cudaMemcpy(hcpy, incpy, R.size() * sizeof(int), cudaMemcpyDeviceToHost);
        
        /*for(int i=0; i<R.size(); i++){
            std::cout<<"hcpy[i]"<<hcpy[i]<<"\n";
        }*/
        Pair* d_result;
        cudaMalloc((void**)&d_result, res * sizeof(Pair));
        
        Pair *result = (Pair *)malloc(res * sizeof(Pair));
        makeAllPairs<<<numBlocks, BLOCK_SIZE>>>(S_ptr, R_ptr, indices, incpy, d_result);
        
        cudaMemcpy(result, d_result, res * sizeof(Pair), cudaMemcpyDeviceToHost);
        /*for(int i=0; i<res; i++){
            std::cout<<"result[i].first"<<result[i].first<<"result[i].second"<<result[i].second<<"\n";
        }*/

    }
    auto end = std::chrono::high_resolution_clock::now();
    cudaFree(indices);
    free(hostRes);

    // Stop measuring time and calculate the elapsed time

    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    printf("Time measured: %.3f ms.\n", elapsed.count() * 1e-6);

    return 0;
}
