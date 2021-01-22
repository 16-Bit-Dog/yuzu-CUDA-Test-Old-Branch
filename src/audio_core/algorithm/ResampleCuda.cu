//to understand data types: http://michas.eu/blog/c_ints.php?lang=en
#include <cuda.h>
#include "device_launch_parameters.h" 
#include <array>
#include "cuda_runtime.h"
#include <cmath>
#include "common/common_types.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/device_free.h> // place holder for testing
#include <thrust/device_malloc.h> // place holder for testing


//global vars are used to force variables into l2 cache when possible; also to prevent variables from deallocating whenever possible

__constant__ s16 lutTd[512];

__global__ void ResampleKernel(s32* output, s32* fraction) {

    std::size_t i = threadIdx.x;
        
    const std::size_t lut_index{ (static_cast<std::size_t>(fraction[i + 1]) >> 8) * 4 }; //fraction is s32, lut_index is size_t


    const s16 l0 = lutTd[lut_index + 0]; //faster this way
    const s16 l1 = lutTd[lut_index + 1];
    const s16 l2 = lutTd[lut_index + 2];
    const s16 l3 = lutTd[lut_index + 3];

    const s32 s0 = fraction[(fraction[i + fraction[0] + 1] + 0 + fraction[0] * 2 + 1)];
    const s32 s1 = fraction[(fraction[i + fraction[0] + 1] + 0 + fraction[0] * 2 + 2)];
    const s32 s2 = fraction[(fraction[i + fraction[0] + 1] + 0 + fraction[0] * 2 + 3)];
    const s32 s3 = fraction[(fraction[i + fraction[0] + 1] + 0 + fraction[0] * 2 + 4)];


    output[i] = (l0 * s0 + l1 * s1 + l2 * s2 + l3 * s3) >> 15;
}

thrust::device_vector <s32> postFractiond; 
thrust::device_vector <s32> outD; 
thrust::host_vector<s32> postFraction; 

extern "C" void ResampleCuda(std::size_t sample_count, s32 * fraction, s32 * output, const s32* input, s32 pitch, const std::array<s16, 512> lut) {
    
    cudaSetDeviceFlags(cudaDeviceLmemResizeToMax); // doing good?
    

    cudaMemcpyToSymbolAsync(lutTd, &lut, sizeof(s16) * (512), 0, cudaMemcpyHostToDevice); //constant memory filled with lut curve values

    
    /* 
    I put together in postFraction in this order for memcpy speed: sample_count [populates index 0], fraction
     values [size of sample_count], index [size of sample_count], input [size of sample_count + 3]   
    */
    
    postFraction.resize(sample_count * 3 + 4); 
    //postFractiond.resize(sample_count * 3 + 4);

    postFraction[1] = *fraction;
    postFraction[sample_count + 1] = 0;
    postFraction[0] = sample_count;

    thrust::copy(input,input+sample_count+3, postFraction.begin()+sample_count * 2 + 1); // copy all 'input' array values

    for (std::size_t i = 1; i < sample_count + 1;
         i++) { 

        postFraction[i + 1] = postFraction[i] + pitch;

        postFraction[i + sample_count + 1] =
            postFraction[i + sample_count] + (postFraction[i + 1] >> 15);

        postFraction[i + 1] &= 0x7fff;
    }
   
    postFractiond = postFraction;

    s32* postFractionP = thrust::raw_pointer_cast(postFractiond.data()); 

    outD.resize(sample_count); //resize premade vector that is in global l2 cache

    s32* outDP = thrust::raw_pointer_cast(outD.data());

    cudaDeviceSynchronize(); // sync up all thread

   // cudaMemPrefetchAsync(postFractionP, sizeof(s32) * (sample_count * 3 + 4), NULL);

   // cudaMemPrefetchAsync(outDP, sizeof(s32) * (sample_count), NULL);

    ResampleKernel <<<1, sample_count>>>(outDP, postFractionP);

    /* KERNEL IS SUPPOSED TO EMULATE THE OPERATION BELOW
    //for (std::size_t i = 0; i < sample_count; i++) {

        const std::size_t lut_index{ (static_cast<std::size_t>(postFraction[i]) >> 8) * 4 }; //fraction is s32, lut_index is size_t

        const s16 l0 = lutH[lut_index + 0]; // s16
        const s16 l1 = lutH[lut_index + 1]; // s16
        const s16 l2 = lutH[lut_index + 2]; // s16
        const s16 l3 = lutH[lut_index + 3]; // s16

        const s32 s0 = (inputH[index[i] + 0]); //s32
        const s32 s1 = (inputH[index[i] + 1]); // s32  
        const s32 s2 = (inputH[index[i] + 2]); // s32
        const s32 s3 = (inputH[index[i] + 3]); // s32

        out[i] = (l0 * s0 + l1 * s1 + l2 * s2 + l3 * s3) >> 15; // output is s32
    */
    
    thrust::copy(outD.begin(), outD.end(),
                 output); // for now thrust::copy seems like the fastest copy operation

}

//burgy kirby ;')
