/*****************************************************************************
! Copyright(C) 2012 Intel Corporation. All Rights Reserved.
!
! The source code, information  and  material ("Material") contained herein is
! owned  by Intel Corporation or its suppliers or licensors, and title to such
! Material remains  with Intel Corporation  or its suppliers or licensors. The
! Material  contains proprietary information  of  Intel or  its  suppliers and
! licensors. The  Material is protected by worldwide copyright laws and treaty
! provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
! modified, published, uploaded, posted, transmitted, distributed or disclosed
! in any way  without Intel's  prior  express written  permission. No  license
! under  any patent, copyright  or  other intellectual property rights  in the
! Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
! implication, inducement,  estoppel or  otherwise.  Any  license  under  such
! intellectual  property  rights must  be express  and  approved  by  Intel in
! writing.
!
! *Third Party trademarks are the property of their respective owners.
!
! Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
! this  notice or  any other notice embedded  in Materials by Intel or Intel's
! suppliers or licensors in any way.
!
!*****************************************************************************
! Content:
! Source component of a simple example of ISO-3DFD implementation. 
! leonardo.borges@intel.com
!****************************************************************************/

#ifndef _TOOLS_INCLUDE
#define _TOOLS_INCLUDE

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <sys/time.h>
#include <math.h>

double walltime() // seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return ((double)(tv.tv_sec) + 1e-06 * (double)(tv.tv_usec));
}

#define dbg_r // printf("for rand : %d\n", __LINE__)

#define UPDATE_SEED(x) ((x) = (x) * 1321 + 59)

void random_data(float *data, const int dimx, const int dimy, const int dimz,
                 const float lower_bound, const float upper_bound)
{
    // printf("%d %d %d\n", dimx, dimy, dimz); 
    char *ctmp = getenv("SEED");
    unsigned int seed=ctmp ? atoi(ctmp) : 0;
    //srand(seed);
    #pragma omp parallel private(seed)
{
    size_t ix, iy, iz; 
    seed += omp_get_thread_num();
    #pragma omp for collapse(3)
    for(iz=0; iz<dimz; iz++)
    for(iy=0; iy<dimy; iy++)
    for(ix=0; ix<dimx; ix++) {
        *(data + iz * dimx * dimy + iy * dimx + ix) = (float) (lower_bound + (UPDATE_SEED(seed)/(float)RAND_MAX) * (upper_bound - lower_bound));
    }
}
    dbg_r;
}

// naive and slow implementation
void reference_3D(float *output, float *input, float *vel, float *coeff, 
                  const int dimx, const int dimy, const int dimz,
                  const int nreps, const int radius)
{
    long dimxy = dimx*dimy;

    int n, ix, iy, iz, ir;
    for(n=0; n<nreps; n++) {
        float *pprev = n%2 ? output : input;
        float *pnext = n%2 ? input : output;
        float *pvel = vel;
        for(iz=0; iz<dimz; iz++) {
            for(iy=0; iy<dimy; iy++) {
                for(ix=0; ix<dimx; ix++) {
                    if( ix>=radius && ix<(dimx-radius) && iy>=radius && iy<(dimy-radius) && iz>=radius && iz<(dimz-radius) ) {
                        float value = 0.0;
                        value += (*pprev)*coeff[0];
                        for(ir=1; ir<=radius; ir++) {
                            value += coeff[ir] * (*(pprev+ir) + *(pprev-ir));	      // horizontal
                            value += coeff[ir] * (*(pprev+ir*(long)dimx) + *(pprev-ir*(long)dimx));   // vertical
                            value += coeff[ir] * (*(pprev+ir*dimxy) + *(pprev-ir*dimxy)); // in front / behind
                        }
                        *pnext = 2.0 * (*pprev) - (*pnext) + value* (*pvel);
                    }
                    ++pnext;
                    ++pprev;
                    ++pvel;
                }
            }
        }
    }    
}

int within_epsilon(float *output, float *reference,
                   const int dimx, const int dimy, const int dimz,
                   const int radius, const int zadjust, const float delta)
{
    int retval = 1;
    float abs_delta = fabsf(delta);
    float maxdiff = 0.0f;
    float minout=output[radius], maxout=output[radius];
    float maxdiffref=reference[radius], maxdiffout=output[radius];
    int ix, iy, iz;
    for(iz=0; iz<dimz; iz++) {
        for(iy=0; iy<dimy; iy++) {
            for(ix=0; ix<dimx; ix++) {
                if( ix>=radius && ix<(dimx-radius) && iy>=radius && iy<(dimy-radius) && iz>=radius && iz<(dimz-radius+zadjust) ) {
                    float difference = fabsf( *reference - *output);
                    if(difference > maxdiff) {
                        maxdiff=difference;
                        maxdiffout=*output;
                        maxdiffref=*reference;
                    }
                    if(*output>maxout) maxout=*output;
                    if(*output<minout) minout=*output;
                    if( difference > delta ) {
                        retval = 0;
                    }
                }
                ++output;
                ++reference;
            }
        }
    }
    printf("MaxDiff: %f/%f(%f); OutputMax: %f, OutputMin: %f.\n", maxdiffout, maxdiffref, maxdiff, maxout, minout);
    return retval;
}

#endif /*_TOOLS_INCLUDE */
