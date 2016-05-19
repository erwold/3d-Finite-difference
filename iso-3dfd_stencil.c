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
! Source component of a simple example of ISO-3DFD implementation
!
!****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "iso-3dfd.h"

/***************************************************************
 *
 * iso_3dfd_stencil: apply 8th order ISO stencil 
 *
 ***************************************************************/
void iso_3dfd_stencil(float *ptr_next_base, float *ptr_prev_base, float *ptr_vel_base, float *coeff,
	       	       const int n1, const int n2, const int n3) { 
  #pragma omp parallel num_threads(NUM_THREADS) 
{
    const int n1n2 = n1*n2;
    const int fullLength = 2*HALF_LENGTH;
    const int vertical_1 = n1, vertical_2 = n1*2, vertical_3 = n1*3, vertical_4 = n1*4;
    const int front_1 = n1n2, front_2 = n1n2*2, front_3 = n1n2*3, front_4 = n1n2*4;
    const float c0=coeff[0], c1=coeff[1], c2=coeff[2], c3=coeff[3], c4=coeff[4];
    const int vertical_5 = n1*5, vertical_6 = n1*6, vertical_7 = n1*7, vertical_8 = n1*8;
    const int front_5 = n1n2*5, front_6 = n1n2*6, front_7 = n1n2*7, front_8 = n1n2*8;
    const float c5=coeff[5], c6=coeff[6], c7=coeff[7], c8=coeff[8];

    const int vertical[HALF_LENGTH] = {vertical_1, vertical_2, vertical_3, vertical_4, 
				       vertical_5, vertical_6, vertical_7, vertical_8
    };
    const int front[HALF_LENGTH] = {front_1, front_2, front_3, front_4
					     , front_5, front_6, front_7, front_8
    };
    int n3End = n3 - HALF_LENGTH;
    int n2End = n2 - HALF_LENGTH;
    int n1End = n1 - HALF_LENGTH;
    register SIMD_TYPE yNextVec, tempVec;
    register SIMD_TYPE beforeVec, afterVec;
    register SIMD_TYPE currentVec;
    SIMD_TYPE velVec, nextVec;
    register SIMD_TYPE upVec;
    register SIMD_TYPE divVec;
    register SIMD_TYPE coeffV;
    register float* ptr_prev;
    float* ptr_next;
    float* ptr_vel;
   
    #if defined(AVX)
      SIMD_TYPE cVec, minusVec, plusVec, tempVec_1, yPrevVec, ySumVec, zNextVec, zPrevVec, zSumVec, tempSumVec, tempVec_3, tempVec_2;     
    #endif

    register size_t offset;
    SIMD_TYPE twoVec = SET_VALUE_INTR(2.0f);
    SIMD_TYPE coeffVec[HALF_LENGTH + 1];
    int i; 
    #pragma noprefetch
    for(i=0; i<=HALF_LENGTH; i++){
       GENERATE_COEFFICIENT_ARRAY_INTR(i)
    }
    
    int bx, by, bz; 
    #ifdef BLOCK_X_Y_Z
      #pragma omp for OMP_SCHEDULE collapse(3)
        for(bx=HALF_LENGTH; bx<n1End; bx+=n1Block){
        for(by=HALF_LENGTH; by<n2End; by+=n2Block){
	for(bz=HALF_LENGTH; bz<n3End; bz+=n3Block){
    #endif

    #ifdef BLOCK_X_Z_Y
      #pragma omp for OMP_SCHEDULE collapse(3)
        for(bx=HALF_LENGTH; bx<n1End; bx+=n1Block){
        for(bz=HALF_LENGTH; bz<n3End; bz+=n3Block){
        for(by=HALF_LENGTH; by<n2End; by+=n2Block){
    #endif

   #ifdef BLOCK_Y_Z_X
      #pragma omp for OMP_SCHEDULE collapse(1)
        for(by=HALF_LENGTH; by<n2End; by+=n2Block){
        for(bz=HALF_LENGTH; bz<n3End; bz+=n3Block){
        for(bx=HALF_LENGTH; bx<n1End; bx+=n1Block){
   #endif

   #ifdef BLOCK_Y_X_Z
      #pragma omp for OMP_SCHEDULE collapse(3)
        for(by=HALF_LENGTH; by<n2End; by+=n2Block){
        for(bx=HALF_LENGTH; bx<n1End; bx+=n1Block){
        for(bz=HALF_LENGTH; bz<n3End; bz+=n3Block){
   #endif

   #ifdef BLOCK_Z_X_Y
      #pragma omp for OMP_SCHEDULE collapse(3)
        for(bz=HALF_LENGTH; bz<n3End; bz+=n3Block){
        for(bx=HALF_LENGTH; bx<n1End; bx+=n1Block){
        for(by=HALF_LENGTH; by<n2End; by+=n2Block){
   #endif

   #ifdef BLOCK_Z_Y_X
      #pragma omp for OMP_SCHEDULE collapse(3)
        for(bz=HALF_LENGTH; bz<n3End; bz+=n3Block){
        for(by=HALF_LENGTH; by<n2End; by+=n2Block){
        for(bx=HALF_LENGTH; bx<n1End; bx+=n1Block){
   #endif

    int izEnd = MIN(bz+n3Block, n3End);
    int iyEnd = MIN(by+n2Block, n2End);
    int ixEnd = MIN(n1Block, n1End-bx);
    
    int ix, iz, iy; 
    for(iz=bz; iz<izEnd; iz++) {
      for(iy=by; iy<iyEnd; iy++) {
	offset = (size_t) iz*n1n2 + iy*n1 + bx;
	ptr_next = ptr_next_base + offset;
	ptr_prev = ptr_prev_base + offset;
	ptr_vel = ptr_vel_base + offset;

	//      #pragma vector nontemporal(ptr_prev)                                                
        #pragma noprefetch
        #pragma ivdep

	for(ix=0;ix<ixEnd; ix+=SIMD_STEP){
          #if defined(MODEL_MIC)
          // x dim          
              SHIFT_MULT_INIT
	      SHIFT_MULT_INTR(1, vertical_1, front_1, coeffVec[1])
	      SHIFT_MULT_INTR(2, vertical_2, front_2, coeffVec[2])
	      SHIFT_MULT_INTR(3, vertical_3, front_3, coeffVec[3])
	      SHIFT_MULT_INTR(4, vertical_4, front_4, coeffVec[4])
	      SHIFT_MULT_INTR(5, vertical_5, front_5, coeffVec[5])
	      SHIFT_MULT_INTR(6, vertical_6, front_6, coeffVec[6])
	      SHIFT_MULT_INTR(7, vertical_7, front_7, coeffVec[7])
	      SHIFT_MULT_INTR(8, vertical_8, front_8, coeffVec[8])
         #elif defined(AVX)
              SHIFT_MULT_INIT
              SHIFT_MULT_INTR(1)
	      SHIFT_MULT_INTR(2)
	      SHIFT_MULT_INTR(3)
	      SHIFT_MULT_INTR(4)
	      SHIFT_MULT_INTR(5)
	      SHIFT_MULT_INTR(6)
	      SHIFT_MULT_INTR(7)
	      SHIFT_MULT_INTR(8)
	      // y/z dim                                                                                                                                     
	      MUL_COEFF_INTR(vertical_1, front_1, coeffVec[1])
	      MUL_COEFF_INTR(vertical_2, front_2, coeffVec[2])
	      MUL_COEFF_INTR(vertical_3, front_3, coeffVec[3])
	      MUL_COEFF_INTR(vertical_4, front_4, coeffVec[4])
	      MUL_COEFF_INTR(vertical_5, front_5, coeffVec[5])
	      MUL_COEFF_INTR(vertical_6, front_6, coeffVec[6])
	      MUL_COEFF_INTR(vertical_7, front_7, coeffVec[7])
	      MUL_COEFF_INTR(vertical_8, front_8, coeffVec[8])
         #endif // avx                                                          

         // update ptr_next & ptr_vel                                         
         REFRESH_NEXT_INTR
       }
      }
     }
    }
   }
  }
 }
} 

void iso_3dfd(float *ptr_next, float *ptr_prev, float *ptr_vel, float *coeff,
	      const int n1, const int n2, const int n3, int nreps) {
  int it;
  for(it=0; it<nreps; it+=2){
    iso_3dfd_stencil ( ptr_next, ptr_prev, ptr_vel, coeff, n1, n2, n3); 
    // here's where boundary conditions+halo exchanges happen

    // Swap previous & next between iterations
    iso_3dfd_stencil ( ptr_prev, ptr_next, ptr_vel, coeff, n1, n2, n3);
  } // time loop
}
