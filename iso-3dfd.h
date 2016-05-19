/*****************************************************************************
! Copyright(C) 2013 Intel Corporation. All Rights Reserved.
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
! Source component of a simple example of ISO-3DFD implementation for 
!   Intel(R) Xeon Phi(TM)
! Version V2.0
! leonardo.borges@intel.com
!****************************************************************************/
#include <omp.h>
#include <immintrin.h>

#define debug

#ifndef _ISO_3DFD_INCLUDE
#define _ISO_3DFD_INCLUDE

/**** Verify results after one ietration? *******/
#define VERIFY_RESULTS
#define HALF_LENGTH 8

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CEILING(X) (X-(int)(X) > 0 ? (int)(X+1) : (int)(X))

#define CACHELINE_BYTES 64

void iso_3dfd(float *next,  float *prev,  float *vel,   float *coeff,
       const int n1, const int n2, const int n3, int nreps);


#define OMP_SCHEDULE schedule(dynamic)

#define C_PREFETCH1(addr, PREF_DIST) _mm_prefetch((char const*) ((addr) + (PREF_DIST)), _MM_HINT_T0);
#define C_PREFETCH2(addr, PREF_DIST) _mm_prefetch((char const*) ((addr) + (PREF_DIST)), _MM_HINT_T1);

#define L2 64
#define L1 16

#if defined(MODEL_MIC)
  #define SIMD_STEP 16
  #define C_PREFETCH(ptr, v2, v1) \
  C_PREFETCH2(ptr, v2) \
  C_PREFETCH1(ptr, v1) 

  #define SET_VALUE_INTR(v) \
  _mm512_set1_ps(v)

  #define GENERATE_COEFFICIENT_ARRAY_INTR(i) \
  C_PREFETCH(&coeff[i], 128, 16) \
  coeffVec[i] = _mm512_set1_ps(coeff[i]);

  #define STORE_YZ_DIMENSION_IN_DIV_INTR \
  C_PREFETCH2(div + ix, L2) \
  C_PREFETCH1(div + ix, L1) \
  _mm512_storenr_ps(div + ix, divVec);

  #define SHIFT_MULT_INTR(ind, vertical, front, coeff) \
  coeffV = coeff; \
  yNextVec = (__m512)_mm512_alignr_epi32((__m512i)currentVec, (__m512i)beforeVec, SIMD_STEP-ind); \
  __assume_aligned((void*)vertical, CACHELINE_BYTES); \
  __assume_aligned((void*)&ptr_prev[ix+vertical], CACHELINE_BYTES);  \
  C_PREFETCH(&ptr_prev[ix + vertical], 32, 16) \
  divVec = _mm512_fmadd_ps(yNextVec, coeffV, divVec); \
  yNextVec = (__m512)_mm512_alignr_epi32((__m512i)afterVec, (__m512i)currentVec, ind); \
  __assume_aligned((void*)&ptr_prev[ix-vertical], CACHELINE_BYTES); \
  C_PREFETCH(&ptr_prev[ix - vertical], 32, 16) \
  divVec = _mm512_fmadd_ps(yNextVec, coeffV, divVec); \
  yNextVec = _mm512_load_ps(&ptr_prev[ix + vertical]); \
  divVec = _mm512_fmadd_ps(yNextVec, coeffV, divVec); \
  yNextVec = _mm512_load_ps(&ptr_prev[ix - vertical]); \
  __assume_aligned((void*)front, CACHELINE_BYTES); \
  __assume_aligned((void*)&ptr_prev[ix+front], CACHELINE_BYTES); \
  C_PREFETCH(&ptr_prev[ix + front], 80, 16) \                       
  divVec = _mm512_fmadd_ps(yNextVec, coeffV, divVec); \            
  yNextVec = _mm512_load_ps(&ptr_prev[ix + front]); \                                 
  __assume_aligned((void*)&ptr_prev[ix-front], CACHELINE_BYTES); \                   
  C_PREFETCH(&ptr_prev[ix - front], 80, 16) \                                              
  divVec = _mm512_fmadd_ps(yNextVec, coeffV, divVec); \
  yNextVec = _mm512_load_ps(&ptr_prev[ix - front]); \                                  
  divVec = _mm512_fmadd_ps(yNextVec, coeffV, divVec); 

  #define SHIFT_MULT_INIT \
  __assume_aligned(&ptr_prev[ix-SIMD_STEP], CACHELINE_BYTES); \
  C_PREFETCH(&ptr_prev[ix-SIMD_STEP], 256, 48) \
  beforeVec = _mm512_load_ps(&ptr_prev[ix-SIMD_STEP]); \
  currentVec = _mm512_load_ps(&ptr_prev[ix]); \
  afterVec = _mm512_load_ps(&ptr_prev[ix+SIMD_STEP]); \
  divVec = _mm512_mul_ps(currentVec, coeffVec[0]);

  #define REFRESH_NEXT_INTR   \
  C_PREFETCH2(&ptr_next[ix], 64) \
  C_PREFETCH1(&ptr_next[ix], 16) \
  nextVec = _mm512_load_ps(&ptr_next[ix]); \
  C_PREFETCH2(&ptr_vel[ix], 64) \
  C_PREFETCH1(&ptr_vel[ix], 16) \
  velVec = _mm512_load_ps(&ptr_vel[ix]); \
  nextVec = _mm512_fmadd_ps(divVec, velVec, _mm512_fmsub_ps(currentVec, twoVec, nextVec)); \
  _mm512_storenrngo_ps(&ptr_next[ix], nextVec);

#elif defined(AVX)
  #define SIMD_STEP 8
  #define C_PREFETCH(ptr, v2, v1) 
 
  #define SET_ZERO_INTR _mm256_setzero_ps()
 
  #define SET_VALUE_INTR(v) _mm256_set1_ps(v)
 
  #define GENERATE_COEFFICIENT_ARRAY_INTR(i) \
  coeffVec[i] = _mm256_set1_ps(coeff[i]);
 
 
  #define STORE_YZ_DIMENSION_IN_DIV_INTR \
  _mm256_store_ps(div + ix, divVec);
 
 
  #define MUL_COEFF_INTR(vertical, front, coeff) \
  __assume_aligned((void*)vertical, CACHELINE_BYTES); \
  __assume_aligned((void*)&ptr_prev[ix+vertical], CACHELINE_BYTES); \
  yNextVec = _mm256_load_ps(&ptr_prev[ix + vertical]); \
  __assume_aligned((void*)&ptr_prev[ix-vertical], CACHELINE_BYTES); \
  yPrevVec = _mm256_load_ps(&ptr_prev[ix - vertical]); \
  ySumVec = _mm256_add_ps(yNextVec, yPrevVec); \
  __assume_aligned((void*)front, CACHELINE_BYTES); \
  __assume_aligned((void*)&ptr_prev[ix+front], CACHELINE_BYTES); \
  zNextVec = _mm256_load_ps(&ptr_prev[ix + front]); \
  __assume_aligned((void*)&ptr_prev[ix-front], CACHELINE_BYTES); \
  zPrevVec = _mm256_load_ps(&ptr_prev[ix - front]); \
  zSumVec = _mm256_add_ps(zNextVec, zPrevVec); \
  tempSumVec = _mm256_add_ps(ySumVec, zSumVec); \
  zSumVec = _mm256_mul_ps(tempSumVec, coeff); \
  divVec = _mm256_add_ps(zSumVec, divVec);
 
  #define SHIFT_MULT_INIT \
  __assume_aligned(&ptr_prev[ix-SIMD_STEP], CACHELINE_BYTES); \
  beforeVec = _mm256_load_ps(&ptr_prev[ix-SIMD_STEP]); \
  currentVec = _mm256_load_ps(&ptr_prev[ix]); \
  afterVec = _mm256_load_ps(&ptr_prev[ix+SIMD_STEP]); \
  divVec = _mm256_mul_ps(currentVec, coeffVec[0]); 
  
  #define SHIFT_MULT_INTR(ind) \
  cVec = coeffVec[ind]; \
  minusVec = _mm256_load_ps(&ptr_prev[ix - ind]); \
  plusVec = _mm256_load_ps(&ptr_prev[ix +ind]); \
  upVec = _mm256_add_ps(minusVec, plusVec); \
  tempVec_1 = _mm256_mul_ps(upVec, cVec); \
  divVec = _mm256_add_ps(tempVec_1, divVec); 
 
  #define REFRESH_NEXT_INTR \
  velVec = _mm256_load_ps(&ptr_vel[ix]); \
  tempVec_3 = _mm256_mul_ps(divVec, velVec);  \
  __assume_aligned((void*)&ptr_next[ix], CACHELINE_BYTES); \
  nextVec = _mm256_load_ps(&ptr_next[ix]); \
  tempVec_1 = _mm256_mul_ps(twoVec, currentVec);  \
  tempVec_2 = _mm256_sub_ps(tempVec_1, nextVec);  \
  nextVec = _mm256_add_ps(tempVec_3, tempVec_2);  \
  _mm256_store_ps(&ptr_next[ix], nextVec);

#endif 

extern int NUM_THREADS;
extern int n1Block;
extern int n2Block;
extern int n3Block; 

#endif /* _ISO_3DFD_INCLUDE */
