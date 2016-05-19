#include <mpi.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#include "iso-3dfd.h"
#include "tools.h"

#define dbg // printf("%d\n", __LINE__)

int n1Block = 256;
int n2Block = 1;
int n3Block = 124;
int NUM_THREADS;

/* Declared to to expose local function from ISO-3DFD previous example */
void iso_3dfd_stencil(float *ptr_next_base,  float *ptr_prev_base,  float *ptr_vel_base,   float *coeff,
                 const int n1, const int n2, const int n3);

#define MASTER 0
#define TAG_UP 0
#define TAG_LW 1
#define TAG_RE 2

#if !defined(CACHELINE_BYTES)
#  define CACHELINE_BYTES   64
#endif
#if !defined(ALIGN_HALO_FACTOR)
#  define ALIGN_HALO_FACTOR  (-HALF_LENGTH )
#endif
#if !defined(MASK_ALLOC_OFFSET)
#  define MASK_ALLOC_OFFSET(x) (x)
#endif
#define NY_TBLOCK 1
#define NZ_TBLOCK 124

/* Syntetic equation cofficients */
float coef[9] = 
{-9.164532312924,
+1.777777777777,
-3.111111111111e-1,
+7.542087542087e-2,
-1.767676767676e-2,
+3.480963480963e-3,
-5.180005180005e-4,
+5.074290788576e-5,
-2.428127428127e-6};

MPI_Request halo_reqs[4]; // Halo async exchange requests 

// Indexes to store halo async exchange requests 
# define reqRecvUP 0
# define reqRecvLW 1   
# define reqSendUP 2
# define reqSendLW 3

// receive_halos: post asynchrnous receive for halos
void receive_halos(float *BuffUP, float *BuffLW,
		   const int BuffSZ,
		   const int rank, const int nranks, MPI_Request *req) {

  const int up = rank-1;
  const int lw = rank+1;

  /* Receive up & below */
  if (rank > 0)
    MPI_Irecv(BuffUP, BuffSZ, MPI_FLOAT, up, TAG_UP, MPI_COMM_WORLD, &req[reqRecvUP]);
  else
    req[reqRecvUP] = MPI_REQUEST_NULL;

  if (rank < (nranks-1))
    MPI_Irecv(BuffLW, BuffSZ, MPI_FLOAT, lw, TAG_LW, MPI_COMM_WORLD, &req[reqRecvLW]);
  else
    req[reqRecvLW] = MPI_REQUEST_NULL;

}

// send_halos: post asynchrnous send for halos
void send_halos(float *HaloUP, float *HaloLW,
		const int BuffSZ,
		const int rank, const int nranks, MPI_Request *req) {

  const int up = rank-1;
  const int lw = rank+1;

  /* Send up & below */

  if (rank < (nranks-1))
    MPI_Isend(HaloLW, BuffSZ, MPI_FLOAT, lw, TAG_UP, MPI_COMM_WORLD, &req[reqSendLW]);
  else
    req[reqSendLW] = MPI_REQUEST_NULL;

  if (rank > 0)
    MPI_Isend(HaloUP, BuffSZ, MPI_FLOAT, up, TAG_LW, MPI_COMM_WORLD, &req[reqSendUP]);
  else
    req[reqSendUP] = MPI_REQUEST_NULL;
}


// wait_exchange_halos: complete all async signals and copy data from buffers to halo slices
void wait_exchange_halos(float *HaloFromUP, float *HaloFromLW,
			 float *BuffUP, float *BuffLW, const int BuffSZ, 
			 const int rank, const int nranks, MPI_Request *req) {
  MPI_Status statuses[4];
  MPI_Waitall(4, req, statuses);
  // This is only an example: here, halos are not copied in parallel
  int i;
  if (rank > 0) {
    #pragma omp parallel for
    for (i=0; i<BuffSZ; i++)
      HaloFromUP[i] = BuffUP[i];
  }

  if (rank < (nranks-1)) {
    #pragma omp parallel for
    for (i=0; i<BuffSZ; i++)
      HaloFromLW[i] = BuffLW[i];
  }
}

void gather_data(float *data, float *my_data, 
        int radius, int nx_ny_pad, int nz,
        int nranks, int *weights, int weighted_nranks) {
    float *data_ptr = data;
    size_t whole_size = nx_ny_pad * nz;
    int r;
    for (r = 0; r < nranks; ++r) {
        size_t size = (whole_size / weighted_nranks) * weights[r];
        MPI_Status status;
        if (r == MASTER) {
            size += radius * nx_ny_pad;
            memcpy(data_ptr, my_data, size * sizeof(float));
        } else if (r < nranks - 1) {
            MPI_Recv(data_ptr, size, MPI_FLOAT, r, 0, MPI_COMM_WORLD, &status);
        } else {
            size += radius * nx_ny_pad;
            MPI_Recv(data_ptr, size, MPI_FLOAT, r, 0, MPI_COMM_WORLD, &status);
        }
        data_ptr += size;
    }
}

void bcast_data(float *data, float *my_data, int radius, int nx_ny_pad, int nz,
        int nranks, int *weights, int weighted_nranks) {
    float *data_ptr = data;
    size_t whole_size = nx_ny_pad * nz;
    int r;
    for (r = 0; r < nranks; ++r) {
        size_t size = (whole_size / weighted_nranks) * weights[r];
        if (r == MASTER) {
            size += radius * nx_ny_pad;
            memcpy(my_data, data_ptr, size * sizeof(float));
        } else if (r < nranks - 1) {
            MPI_Send(data_ptr, size, MPI_FLOAT, r, 0, MPI_COMM_WORLD);
        } else {
            size += radius * nx_ny_pad;
            MPI_Send(data_ptr, size, MPI_FLOAT, r, 0, MPI_COMM_WORLD);
        }
        data_ptr += size;
    }
}

void initialize_memory(int nx_ny_pad, int proc_nz_pad,
        float **p_proc_area_u0, float **p_proc_area_u1, float **p_vel) {
  size_t size_proc_area = (size_t) proc_nz_pad * (size_t) nx_ny_pad; 

  float *proc_area_u0_base = (float *) _mm_malloc( (size_proc_area+16+MASK_ALLOC_OFFSET(0 ))*sizeof(float), CACHELINE_BYTES);    
  float *proc_area_u1_base = (float*) _mm_malloc( (size_proc_area+16+MASK_ALLOC_OFFSET(16))*sizeof(float), CACHELINE_BYTES);
  float *vel_base = (float*) _mm_malloc( (size_proc_area+16+MASK_ALLOC_OFFSET(32))*sizeof(float), CACHELINE_BYTES);

  dbg;
  if(!proc_area_u0_base || !proc_area_u1_base || !vel_base) {
      puts("can not allocate memory");
      exit(1);
  }

  dbg;
  memset(proc_area_u0_base, 0, (size_proc_area+16+MASK_ALLOC_OFFSET(0))*sizeof(float));  
  memset(proc_area_u1_base, 0, (size_proc_area+16+MASK_ALLOC_OFFSET(16))*sizeof(float));
  memset(vel_base,          0, (size_proc_area+16+MASK_ALLOC_OFFSET(32))*sizeof(float));  

  // Align working vectors offsets 
  *p_proc_area_u0 = &proc_area_u0_base[16 +ALIGN_HALO_FACTOR +MASK_ALLOC_OFFSET(0 )];
  *p_proc_area_u1 = &proc_area_u1_base[16 +ALIGN_HALO_FACTOR +MASK_ALLOC_OFFSET(16)];
  *p_vel          = &vel_base         [16 +ALIGN_HALO_FACTOR +MASK_ALLOC_OFFSET(32)];
}

void free_memory(int nx_ny_pad, int proc_nz_pad,
        float *proc_area_u0, float *proc_area_u1, float *vel) {
  size_t size_proc_area = (size_t) proc_nz_pad * (size_t) nx_ny_pad; 
  float *proc_area_u0_base = proc_area_u0 - (16+ALIGN_HALO_FACTOR +MASK_ALLOC_OFFSET(0 ));
  float *proc_area_u1_base = proc_area_u1 - (16+ALIGN_HALO_FACTOR +MASK_ALLOC_OFFSET(16));
  float *vel_base          = vel          - (16+ALIGN_HALO_FACTOR +MASK_ALLOC_OFFSET(32));

  _mm_free(proc_area_u0_base);
  _mm_free(proc_area_u1_base);
  _mm_free(vel);
}


void iso_3dfd_mpi(float *proc_area_u1,  float *proc_area_u0, float *vel, float *coeff,
                 const int nx_pad, const int ny_pad, const int proc_nz, const int nsteps,
                 const int radius, const int rank, const int nranks) {
  const size_t nx_ny_pad = nx_pad * ny_pad;
  const size_t guard_slice_UP   = 0;                         // const int sz_guard_slice_UP   = radius;
  const size_t hallo_compute_UP = radius*nx_ny_pad;             const int sz_hallo_compute_UP = 3*radius;
  const size_t hallo_from_UP    = radius*nx_ny_pad;          // const int sz_hallo_from_UP    = radius;
  const size_t hallo_slice_UP   = 2*radius*nx_ny_pad;        // const int sz_hallo_slice_UP   = radius;
  const size_t true_inner       = 2*radius*nx_ny_pad;           const int sz_true_inner       = proc_nz;
  const size_t hallo_compute_LW = proc_nz*nx_ny_pad;            const int sz_hallo_compute_LW = 3*radius;
  const size_t hallo_slice_LW   = (proc_nz+radius)*nx_ny_pad;// const int sz_hallo_slice_LW   = radius;
  const size_t hallo_from_LW    = (proc_nz+2*radius)*nx_ny_pad;//const int sz_hallo_from_LW    = radius;
  const size_t guard_slice_LW   = (proc_nz+3*radius)*nx_ny_pad; const size_t sz_guard_slice_LW   = radius*nx_ny_pad;

  receive_halos(&proc_area_u0[guard_slice_UP], &proc_area_u0[guard_slice_LW],
          sz_guard_slice_LW, rank, nranks, halo_reqs);

  send_halos(&proc_area_u0[hallo_slice_UP], &proc_area_u0[hallo_slice_LW],
          sz_guard_slice_LW, rank, nranks, halo_reqs);

  wait_exchange_halos(&proc_area_u0[hallo_from_UP], &proc_area_u0[hallo_from_LW],
          &proc_area_u0[guard_slice_UP], &proc_area_u0[guard_slice_LW],
          sz_guard_slice_LW, rank, nranks, halo_reqs);
 
  dbg;
  int istep;
  for (istep = 0; istep < nsteps; istep+=1) {
    // Change asynchronously the upper and lower halos, needed at the next timestep.
    receive_halos(&proc_area_u1[guard_slice_UP], &proc_area_u1[guard_slice_LW],
		  sz_guard_slice_LW,
		  rank, nranks, halo_reqs);
	
    dbg;
    // ISO3DFD- the upper: proc_area_u1[UP] <- proc_area_u0
    iso_3dfd_stencil(&proc_area_u1[hallo_compute_UP], &proc_area_u0[hallo_compute_UP],  &vel[hallo_compute_UP], coef,
		nx_pad, ny_pad, sz_hallo_compute_UP);

    // ISO3DFD the lower: proc_area_u1[LW] <- proc_area_u0
    iso_3dfd_stencil(&proc_area_u1[hallo_compute_LW], &proc_area_u0[hallo_compute_LW],  &vel[hallo_compute_LW], coef,
		nx_pad, ny_pad, sz_hallo_compute_LW);

    // Change asynchronously the upper and lower halos, needed at the next timestep.
    send_halos(&proc_area_u1[hallo_slice_UP], &proc_area_u1[hallo_slice_LW],
	       sz_guard_slice_LW,
	       rank, nranks, halo_reqs);

    dbg;
    // ISO3DFD the core
    // Check if there are more z's to do.
    if (sz_true_inner > 0) {
      iso_3dfd_stencil(&proc_area_u1[true_inner], &proc_area_u0[true_inner],  &vel[true_inner], coef,
		  nx_pad, ny_pad, sz_true_inner);
    }
	
    dbg;
    // Wait for the borders
    wait_exchange_halos(&proc_area_u1[hallo_from_UP], &proc_area_u1[hallo_from_LW],
			&proc_area_u1[guard_slice_UP], &proc_area_u1[guard_slice_LW],
			sz_guard_slice_LW, rank, nranks, halo_reqs);

    dbg;
    // Swap vectors
    float *proc_area_tmp = proc_area_u0;
    proc_area_u0 = proc_area_u1;
    proc_area_u1 = proc_area_tmp;
  }
}

int main (int argc, char* argv[]) {
  int rank;
  char hostname[256];
  float * proc_area_u0;
  float * proc_area_u1;
  float * vel;

  double wstart, wstop;
  float elapsed_time=0.0f, throughput_mpoints=0.0f, mflops=0.0f; 

  // Deafults
  int nx, ny, nz, nx_pad, ny_pad, nz_pad, nx_ny_pad, nx_ny_nz_pad, proc_nz, proc_nz_pad;
  int nsteps;
  int nx_Tblock = nx;
  int ny_Tblock = NY_TBLOCK;
  int nz_Tblock = NZ_TBLOCK;
  #pragma omp parallel
  { NUM_THREADS = omp_get_num_threads(); }

  /* This is used to balance workload between xeon phi and xeon. The more faster is phi more slices of z it will receive. */
  int factor_speed_xphi = 20;
  int factor_speed_xeon = 10; 

  if( argc < 5 ) {
    printf(" usage: [nx ny nz threads] [# iterations] [x-block] [y-block] [z-block] [Phi speed] [CPU speed]\n");
    exit(1);
  }

  // [n1 n2 n3 # iterations]
  nx  = atoi(argv[1]); nx = (nx /16 + 1)*16; // adjust x
  ny  = atoi(argv[2]);
  nz  = atoi(argv[3]);
  nsteps = atoi(argv[4]);
  // [x-block] [y-block] [z-block]
  if( argc >= 6)
    n1Block = atoi(argv[5]);
  if( argc >= 7)
    n2Block = atoi(argv[6]);
  if( argc >= 8)
    n3Block = atoi(argv[7]);
  // [Phi speed] [CPU speed]
  if( argc >= 9)
    factor_speed_xphi = atoi(argv[8]);
  if( argc >= 10)
    factor_speed_xeon = atoi(argv[9]);

  int nranks;
  MPI_Init (&argc, &argv);			/* starts MPI */
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &nranks);	/* get number of processes */

  // Summation of all MPI ranks with respective Xeon Phi / CPU performance weights
# ifdef __MIC__
  int my_weight=factor_speed_xphi;
# else
  int my_weight=factor_speed_xeon;
# endif

  int weighted_nranks;
  int *weights;

  if (rank == MASTER)
      weights = (int *) malloc(nranks * sizeof(int));
  MPI_Gather (&my_weight, 1, MPI_INT, weights, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Allreduce (&my_weight, &weighted_nranks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  gethostname(hostname, sizeof(hostname));
  printf( "Process %d of %d on %s: my_weight=%d, weighted_nranks=%d\n", rank, nranks, hostname, my_weight, weighted_nranks);

  const int radius = HALF_LENGTH;

#if(1)
  nz = (nz / weighted_nranks + 1) * weighted_nranks; // adjust z

  nx_pad = nx+2*radius;
  ny_pad = ny+2*radius;

  nx_ny_pad = nx_pad * ny_pad;

  // Length of the slice in Z dimention to be computed by this MPI rank.
  proc_nz = (nz / weighted_nranks) * my_weight;
  proc_nz_pad = proc_nz + 4*radius;

  initialize_memory(nx_ny_pad, proc_nz_pad, &proc_area_u0, &proc_area_u1, &vel);
  dbg;

  random_data(proc_area_u0, nx_pad, ny_pad, proc_nz_pad, 0.2, 0.7);
  random_data(proc_area_u1, nx_pad, ny_pad, proc_nz_pad, -0.2, 0.2);
  random_data(vel, nx_pad, ny_pad, proc_nz_pad, 0.01, 0.02);

  printf("[%d] Time steps.\n", rank);
  MPI_Barrier(MPI_COMM_WORLD); 

  if(rank == MASTER || nranks == 1) {
    wstart = walltime();
  }

  iso_3dfd_mpi(proc_area_u1, proc_area_u0, vel, coef, nx_pad, ny_pad, proc_nz, nsteps,
          radius, rank, nranks);

  // End processing
  MPI_Barrier(MPI_COMM_WORLD); 

  // Master collects stats
  if(rank == MASTER || nranks == 1) {
    wstop =  walltime();

    // Report time and performance
    elapsed_time = wstop - wstart;
    float normalized_time = elapsed_time/nsteps;   
    // throughput_mpoints = ((size_t)(nx-2*radius)*(ny-2*radius)*(nz-2*radius))/(normalized_time*1e6f);
    throughput_mpoints = ((size_t)nx*ny*nz)/(normalized_time*1e6f);
    mflops = (7.0f*radius + 5.0f)* throughput_mpoints;

    printf("-------------------------------\n");
    printf("time:       %8.2f sec\n",       elapsed_time );
    printf("throughput: %8.2f MPoints/s\n", throughput_mpoints );
    printf("flops:      %8.2f GFlops\n",     mflops/1e3f );
  }

  free_memory(nx_ny_pad, nz_pad, proc_area_u0, proc_area_u1, vel);
#endif

#define VERIFY_RESULTS

#if defined(VERIFY_RESULTS)
  nx = 1888; nx = (nx /16 + 1)*16; // adjust x
  ny = 1111;
  nz = 229; nz = (nz / weighted_nranks + 1) * weighted_nranks; // adjust z
  nsteps = 16;
  n1Block = 1888;
  n2Block = 1;
  n3Block = 128;

  nx_pad = nx + 2 * radius;
  ny_pad = ny + 2 * radius;
  nz_pad = nz + 2 * radius;
  nx_ny_pad = nx_pad * ny_pad;
  nx_ny_nz_pad = nx_ny_pad * nz_pad;

  proc_nz = (nz / weighted_nranks) * my_weight;
  proc_nz_pad = proc_nz + 4 * radius;
  initialize_memory(nx_ny_pad, proc_nz_pad, &proc_area_u0, &proc_area_u1, &vel);

  // -------- data broadcast
  float *all_area_u1;
  float *all_area_u0;
  float *all_vel;
  if (rank == MASTER) {
    all_area_u1 = (float *) malloc(nx_ny_nz_pad * sizeof(float));
    all_area_u0 = (float *) malloc(nx_ny_nz_pad * sizeof(float));
    all_vel = (float *) malloc(nx_ny_nz_pad * sizeof(float));

    random_data(all_area_u0, nx_pad, ny_pad, nz_pad, 0.2, 0.7);
    random_data(all_area_u1, nx_pad, ny_pad, nz_pad, -0.2, 0.2);
    random_data(all_vel, nx_pad, ny_pad, nz_pad, 0.01, 0.02);

    bcast_data(all_area_u1, &proc_area_u1[radius * nx_ny_pad], radius, nx_ny_pad, nz,
            nranks, weights, weighted_nranks);
    bcast_data(all_area_u0, &proc_area_u0[radius * nx_ny_pad], radius, nx_ny_pad, nz,
            nranks,  weights, weighted_nranks);
    bcast_data(all_vel, &vel[radius * nx_ny_pad], radius, nx_ny_pad, nz,
            nranks, weights, weighted_nranks);
  }
  else {
      int size = proc_nz * nx_ny_pad;
      if (rank == nranks - 1)
          size += radius * nx_ny_pad;
      MPI_Status status;
      MPI_Recv(&proc_area_u1[2*radius*nx_ny_pad], size, MPI_FLOAT, MASTER,
              0, MPI_COMM_WORLD, &status);
      MPI_Recv(&proc_area_u0[2*radius*nx_ny_pad], size, MPI_FLOAT, MASTER,
              0, MPI_COMM_WORLD, &status);
      MPI_Recv(&vel[2*radius*nx_ny_pad], size, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD, &status);
  }
  // -------- end of data broadcast

  MPI_Barrier(MPI_COMM_WORLD); 

  iso_3dfd_mpi(proc_area_u1, proc_area_u0, vel, coef, nx_pad, ny_pad, proc_nz, nsteps,
          radius, rank, nranks);

  MPI_Barrier(MPI_COMM_WORLD); 

  if (rank == MASTER) {
    float *all_result = (float *) malloc(nx_ny_nz_pad * sizeof(float));
    gather_data(all_result, &proc_area_u1[radius * nx_ny_pad], radius, nx_ny_pad, nz,
            nranks, weights, weighted_nranks);
    reference_3D(all_area_u1, all_area_u0, all_vel, coef,
            nx_pad, ny_pad, nz_pad, nsteps, HALF_LENGTH);
    if(within_epsilon(all_result, all_area_u1, nx_pad, ny_pad, nz_pad, HALF_LENGTH, 0, 0.00005f)) {
        printf("  Result within epsilon\n");
        printf("  TEST PASSED!\n");
    } else {
        printf("  Incorrect result\n");	
        printf("  TEST FAILED!\n");
    }
  } else {
      int size = proc_nz * nx_ny_pad;
      if (rank == nranks - 1)
          size += radius * nx_ny_pad;
      MPI_Send(&proc_area_u1[2*radius*nx_ny_pad], size, MPI_FLOAT, MASTER,
              0, MPI_COMM_WORLD);
  }
#endif

  MPI_Finalize();
  return 0;
} 
