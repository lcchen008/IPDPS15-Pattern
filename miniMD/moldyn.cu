/************************************************************
!File     : moldyn.c
!Rewritten by Hwansoo Han
!
!Description :  Calculates the motion of  particles
!               based on forces acting on each particle
!               from particles within a certain radius. 
!
!Contents : The main computation is in function main()
!           The structure of the computation is as follows:
!
!     1. Initialise variables
!     2. Initialise coordinates and velocities of molecules based on
!          some distribution.
!     3. Iterate for N time-steps
!         3a. Update coordinates of molecule.
!         3b. On Every xth iteration 
!             ReBuild the interaction-list. This list
!             contains every pair of molecules which 
!             are within a cutoffSquare radius  of each other.
!         3c. For each pair of molecule in the interaction list, 
!             Compute the force on each molecule, its velocity etc.
!     4.  Using final velocities, compute KE and PE of system.
!
!Input Data :
!      The default setting simulates the dynamics with 8788
!      particles. A smaller test-setting can be achieved by
!      changing  BOXSIZE = 4.  To do this, change the #undef SMALL
!      line below to #define SMALL. No other change is required.
*************************************************************/

#define EXTERN

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include "moldyn.h"
//#include "misc.h"
#include "irregular/irregular_runtime.h"
#include "lib/time_util.h"
#include <omp.h>
#include "include/runtime.h"
#include "regular/regular_runtime.h"
//#include "regular.h"
//#include "reorder.cpp"

#ifdef TIME
#include "timer.h"
#endif

#define NPROC 4 

extern int LEVEL_arg;
extern int DIMENSION;
extern int npart_arg;
extern int nPass, nShift;
extern int LR_param_override;
extern int LR_apply_cnt;

extern "C"
void ReadInputGraph(char *filename, int *n_nodes, int *n_edges, int el[][2]);
extern "C"
void ReadCoord(char *filename, int n_nodes, double xyz[][3]);

/*
!============================================================================
!  Function : main()
!  Purpose  :  
!      All the main computational structure  is here
!      Iterates for specified number of  timesteps.
!      In each time step, 
!        UpdateCoordinates() changes molecules coordinates based
!              on the velocity of that molecules
!              and that molecules
!        BuildNeigh() rebuilds the interaction-list on
!              certain time-steps
!        Computeorces() - the time-consuming step, iterates
!              over all interacting pairs and computes forces
!        UpdateVelocities() - updates the velocities of
!              all molecules  based on the forces. 
!============================================================================
*/

#define CONFLICT 
#ifdef CONFLICT 
double  v [NUM_PARTICLES+4][3];  /* x,y,z coordinates of each molecule */

float  f [NUM_PARTICLES+4][3];  /* partial forces on each molecule    */
#else
double  v [NUM_PARTICLES][3];  /* x,y,z coordinates of each molecule */

double  f [NUM_PARTICLES][3];  /* partial forces on each molecule    */
#endif

double  vh [NUM_PARTICLES][3];   /* partial x,y,z velocity of molecule */

/* (inter1, inter2) stores pairs of interacting molecules */
int     inter [MAXINTERACT][2]; 

/**************/

#define vhx(i)	(vh[(i)][0])
#define vhy(i)	(vh[(i)][1])
#define vhz(i)	(vh[(i)][2])

#define x(i)	(v[(i)][0])
#define y(i)	(v[(i)][1])
#define z(i)	(v[(i)][2])

#define fx(i)	(f[(i)][0])
#define fy(i)	(f[(i)][1])
#define fz(i)	(f[(i)][2])


#define inter1(i) (inter[(i)][0])
#define inter2(i) (inter[(i)][1])

/**************/

double   side,                  /*  length of side of box                 */
         sideHalf,              /*  1/2 of side                           */
         cutoffRadius,          /*  cuttoff distance for interactions     */
         perturb,               /*  perturbs initial coordinates          */
         timeStep,              /*  length of each timestep   */
         timeStepSq,            /*  square of timestep        */
         timeStepSqHalf,        /*  1/2 of square of timestep */
         vaver;                 /*                            */
double   epot,                  /*  The potential energy      */
         vir;                   /*  The virial  energy        */

int      ninter;                /*  number of interacting molecules pairs */

/*
#define   numTimeSteps  NTIMESTEP 
#define   numMoles      (4*BOXSIZE*BOXSIZE*BOXSIZE)
#define   neighUpdate   (10*(1+SCALE_TIMESTEP/4))
*/
int	  numTimeSteps;
int	  numMoles;
int 	  boxSize;
int	  neighUpdate;

struct _dsize_ {
	int     nB;	/* boxSize */
	int	nTS;	/* num TimeSteps */
	int	nU;	/* neighbor Update rate */
} dsize;

#ifdef BLOCK_ADAPTIVE
void RebuildInteraction(double (*)[3], int, double, double);
int  sort_edgelist( int, int, int (*)[2]);
#endif

void RebuildNeighbor();

main(int argc, char **argv)
{
  double   count, vel ;
  double   ekin, percent;
  int      tmp,procs,i,j,k,ii, start_time;
  int      tstep, n_tstep, n_moles, n_inter;
  char     filename1[30], filename2[30];
  int      update_freq = 1, naive = 0;
  int      start, end, how_often;
  double cutoffSquare;
  //double xx, yy, zz, rd; //rrd, rrd2, rrd3, rrd4,  rrd6, rrd7, r148;
  //double forcex, forcey, forcez;
  double sum;
  double vaverh, velocity, counter, sq;
  double reduction_time = 0; 
  double copy_time = 0;
#ifdef TIME
  int      nupdate = 0;
  double   init_time, prev_time = 0.0, tmp_time, update_time = 0.0;
#endif

  if (argc == 2) {  
      if (argv[1][0] == 'n') {
	  sscanf(&(argv[1][1]), "%d", &naive); 
      }
      else { /* for RCB, KMETIS */
          sscanf(argv[1], "%d", &LEVEL_arg);
          npart_arg = 1;
          npart_arg <<= LEVEL_arg;
          printf("[L,P]=[%d, %d]\n", LEVEL_arg, npart_arg);
	  LR_param_override = 1;
      }
  }

  scanf("%d", &dsize.nB); 
  scanf("%d", &dsize.nTS); 
  scanf("%d", &dsize.nU); 
  scanf("%lf", &cutoffRadius);
  scanf("%s", filename1);
  scanf("%s", filename2);
  scanf("%lf", &percent);
  scanf("%d", &start_time);
  scanf("%d", &update_freq);

  boxSize = dsize.nB;
  numMoles = 4 * boxSize * boxSize * boxSize;
  numTimeSteps = dsize.nTS;
  neighUpdate = dsize.nU;

  printf("\n    numMoles = %d  numTimeSteps = %d  neighUpdate = %d  cutoffRadius = %.2lf\n\n",
	   numMoles, dsize.nTS, dsize.nU, cutoffRadius); 

  if (numMoles > NUM_PARTICLES) 
	printf ("numMoles exceeds MAX(%d)\n", NUM_PARTICLES);

    /*........................................................*/

     printf ("Reading Graph...\n");
     ReadInputGraph(filename1, &numMoles, &ninter, inter);

     printf("first inter: (%d %d)\n", inter[0][0], inter[0][1]);

     printf ("Reading Coord...\n");
     ReadCoord(filename2, numMoles, v);

     InitSettings   ();
/*   InitCoordinates(); */
     InitVelocities ();
     InitForces     (); 
  
    /*........................................................*/

     
     //int cpu_partitions[3];
     //cpu_partitions[0] = CPU_X;
     //cpu_partitions[1] = CPU_Y;
     //cpu_partitions[2] = CPU_Z;
     //int gpu_partitions[3];
     //gpu_partitions[0] = GPU_X;
     //gpu_partitions[1] = GPU_Y;
     //gpu_partitions[2] = GPU_Z;

     //struct partition_args partitions;
     //partitions.cpu_gpu_split_point = NUM_POINTS/2;
     //partitions.cpu_partition = cpu_partitions;
     //partitions.gpu_partition = gpu_partitions;

     //Partitioner partitioner(NUM_POINTS, NUM_EDGES, 3, &partitions);  

     //printf("reordering points...\n");

     //partitioner.begin_partition(&v[0][0]);

     //double beforebuild = rtclock();
     //RebuildNeighbor();
     //double afterbuild = rtclock();

     //printf("build time: %f\n", afterbuild - beforebuild);

     //printf("reordering and partitioning edges...\n");

     //double beforereorder = rtclock();
     //partitioner.reorder_edges(&inter[0][0]);

     //int cpu_total_edges = partitioner.get_cpu_num_edges();
     //int gpu_total_edges = partitioner.get_gpu_num_edges();

     //printf("getting edges...\n");
     //struct edge *cpu_edges = partitioner.get_cpu_edges();
     //struct edge *gpu_edges = partitioner.get_gpu_edges();
     
     //printf("***cpu edges: %d gpu edges: %d***\n", cpu_total_edges, gpu_total_edges);
     double afterreorder = rtclock();

     //printf("reordering velocity...\n");
     //partitioner.reorder_satellite_data((double3 *)(&vh[0][0]));
     //printf("reorder time: %f\n", afterreorder - beforereorder);

     //Part *cpu_parts = partitioner.get_cpu_parts();
     //Part *gpu_parts = partitioner.get_gpu_parts();

     //printf("*****printing cpu partitition details***\n");

     //for(int i = 0; i < )

     //int *part_index = partitioner.get_part_index();

#ifdef TIME
     init_time = get_timer();
#endif

     printf("Start iteration...\n");
     n_tstep = numTimeSteps;
     n_moles = numMoles;
     struct point_data *data_h = (struct point_data *)malloc(sizeof(struct point_data)*NUM_POINTS);

     double before = rtclock();
     //data_h->cutoffRadius = cutoffRadius;
     //data_h->side = side;
     //memcpy(data_h->part_index, part_index, NUM_POINTS*sizeof(int));
     memcpy(data_h, &v[0][0], NUM_PARTICLES*sizeof(double)*3); //v is coordinates
     double after = rtclock();

     struct para pa;

     pa.side = side;
     pa.cutoffRadius = cutoffRadius;

     copy_time += (after - before);

     RuntimeInit(argc, argv);

     IrregularRuntime runtime
     (	NUM_EDGES,
	NUM_POINTS,
	(EDGE *)&inter[0][0],
	NULL,
	data_h,
	0,
	sizeof(struct point_data),
	sizeof(double)*3,
	3,
	&v[0][0],	
	sizeof(double),
	&pa,
	sizeof(struct para),
	NPROC,
	1
     );

     runtime.IrregularInit();
     
     for (tstep = 0; tstep < 10/*n_tstep*/; tstep++) 
     //for (tstep = 0; tstep < 1; tstep++) 
     {
     	printf("iteration: %d\n", tstep);

        /*................*/
        /* UpdateCoordinates(); */
	//update coordinates
        for ( i=0; i<n_moles; i++) 
	{
          x(i) = x(i) + vhx(i) + fx(i);
          y(i) = y(i) + vhy(i) + fy(i);
          z(i) = z(i) + vhz(i) + fz(i);

          if ( x(i) < 0.0 )  x(i) = x(i) + side ; 
          if ( x(i) > side ) x(i) = x(i) - side ;
          if ( y(i) < 0.0 )  y(i) = y(i) + side ;
          if ( y(i) > side ) y(i) = y(i) - side ;
          if ( z(i) < 0.0 )  z(i) = z(i) + side ;
          if ( z(i) > side ) z(i) = z(i) - side ;

          vhx(i) = vhx(i) + fx(i);
          vhy(i) = vhy(i) + fy(i);
          vhz(i) = vhz(i) + fz(i);
          fx(i)  = 0.0;
          fy(i)  = 0.0;
          fz(i)  = 0.0;
        }

	if (tstep == start_time) 
	{
	#ifdef TIME
	     init_time = get_timer();
	#endif
	}

	printf("coordinate updated...\n");

        /*................*/
        /* ComputeForces(); */
	
	//copy coordinates and forces

	double before_copy = rtclock();
	memcpy(data_h, v, NUM_PARTICLES*sizeof(double)*3);
	runtime.reset_node_data(data_h);
	double after_copy = rtclock();

	printf("data_h reset++++++++++++++++\n");

	copy_time += (after_copy - before_copy);

	double before_reduction = rtclock();
	runtime.IrregularStart();
	double after_reduction = rtclock();

	if(tstep >= 2)
	reduction_time += (after_reduction - before_reduction);

	//copy back data to host
	runtime.get_reduction_result(&f[0][0]);

        ///*................*/
        ///* UpdateVelocities(); */
        for (int i = 0; i< n_moles; i++) {
          fx(i)  = fx(i) * timeStepSqHalf ;
          fy(i)  = fy(i) * timeStepSqHalf ;
          fz(i)  = fz(i) * timeStepSqHalf ;
          vhx(i) += fx(i);
          vhy(i) += fy(i);
          vhz(i) += fz(i);
        }

	printf("+O+O+O+ finish %d iteration. continuw...+O+O+O+ \n", tstep);

	runtime.IrregularBarrier();
      } 
#ifdef TIME
      printf("Time = %lf | Update_Time = %lf (applied=%d)\n", 
		get_timer()-init_time, update_time, LR_apply_cnt);
#endif

      /*........................................................*/

      /*................*/
      /* ComputeKE      (&ekin); */
      
      double beforeser = rtclock();
      sum = 0.0 ;
      for ( i = 0; i< n_moles; i++) {
        sum = sum + vhx( i ) * vhx( i );
        sum = sum + vhy( i ) * vhy( i );
        sum = sum + vhz( i ) * vhz( i );
      }
      double afterser = rtclock();

      printf("SERIAL TIME: ~~~~ %f\n", afterser - beforeser);

      //ekin = sum/timeStepSq ;

      /* .............................. */
      /* ComputeAvgVel  (&vel, &count); */

      vaverh = vaver * timeStep ;
      velocity    = 0.0 ;
      counter     = 0.0 ;

double beforeave = rtclock();

      for (i=0; i<n_moles; i++) {
         sq = SQRT(  SQR(vhx(i)) + SQR(vhy(i)) +
                     SQR(vhz(i))  );
         if ( sq > vaverh ) counter += 1.0 ;
         velocity += sq ;
      }

double afterave = rtclock();

//generate offsets
      Offset *offsets = (Offset *)malloc(sizeof(Offset) * n_moles);
      for(int i = 0; i < n_moles; i++)
      {
     	offsets[i].offset = i * 3 * sizeof(double); 
	offsets[i].size = sizeof(double) * 3;
      }

      float para = vaverh;

      RegularRuntime rr(NPROC, &vh[0][0], sizeof(double)*3*n_moles, offsets, n_moles, &para, sizeof(float));

      rr.RegularInit();

      rr.RegularStart();


printf("AVE~~~~~~~~~~%f\n", afterave - beforeave);


      vel = (velocity/timeStep);
      count = counter;

      PrintResults ( tstep, ekin, vel, count); 

      printf("reduction time: %f, copy time: %f\n", reduction_time, copy_time);

      RuntimeFinalize();
      /*........................................................*/
      /* PrintStats(); */
}
    


/*---------- INITIALIZATION ROUTINES HERE               ------------------*/

/*
!============================================================================
!  Function :  InitSettings()
!  Purpose  : 
!     This routine sets up the global variables
!============================================================================
*/

void InitSettings()
{
   int i, org_neighUpdate;
   double org_cutoff;

   printf("Init Settings ...\n");
   side   = POW( ((double)(numMoles)/DENSITY), 0.3333333);
   sideHalf  = side * 0.5 ;

   /* cutoffRadius  = MIN(CUTOFF, sideHalf ); */
   org_cutoff  = MIN(CUTOFF, sideHalf ); 

   timeStep      = DEFAULT_TIMESTEP/SCALE_TIMESTEP ;
   timeStepSq    = timeStep   * timeStep ;
   timeStepSqHalf= timeStepSq * 0.5 ;
   org_neighUpdate  =  (10*(1+SCALE_TIMESTEP/4));

   perturb       = side/ (double)boxSize;     /* used in InitCoordinates */
   vaver         = 1.13 * SQRT(TEMPERATURE/24.0);


#ifdef VERBOSE
   printf("----------------------------------------------------");
   printf("\n MolDyn - A Simple Molecular Dynamics simulation \n");
   printf("----------------------------------------------------");
   printf("\n number of particles is ......... %6d", numMoles);
   printf("\n side length of the box is ...... %13.6lf",side);
   printf("\n cut off radius is .............. %13.6lf",cutoffRadius);
   printf("\n temperature is ................. %13.6lf",TEMPERATURE);
   printf("\n time step is ................... %13.6lf",timeStep);
   printf("\n interaction-list updated every..   %d steps",neighUpdate);
   printf("\n total no. of steps .............   %d ",numTimeSteps);
   printf("\n\n Relax, This will take a while. \n\n"); 
   printf(
     "\n TimeStep   K.E.        P.E.        Energy    Temp.     Pres.    Vel.    rp ");
   printf(
     "\n -----    --------   ----------   ----------  -------  -------  ------  ------");
 
#else
  printf("\n   Temperature=%.3lf timeStep=%.3lf perturb=%.3lf side=%lf\n", 
	TEMPERATURE, timeStep, perturb, side);
  printf("   cutoff(org)=%.3lf Update(org)=%d\n\n",org_cutoff,org_neighUpdate);
#endif /* VERBOSE */

#ifdef SCRATCH
 scratch_data += sizeof(double)*(NUM_PARTICLES) ;
#endif
}

/* free(owner);
!============================================================================
!  Function : InitCoordinates()
!  Purpose  :
!     Initialises the coordinates of the molecules by 
!     distribuuting them uniformly over the entire box
!     with slight perturbations.
!============================================================================
*/

void RebuildNeighbor()
{
	   double cutoffSquare = (cutoffRadius * TOLERANCE)*(cutoffRadius * TOLERANCE);
	   ninter = 0;

	   //omp_set_num_threads(4);

	   //#pragma omp parallel for (private j)
	   int i, j;
	   double xx, yy, zz;
	   double rd;
	   for ( i=0; i< numMoles; i++) 
	   {
	     for ( j = i+1; j< numMoles; j++ ) 
	     {

		 xx = x(i) - x(j);
		 yy = y(i) - y(j);
		 zz = z(i) - z(j);

		 if ( xx < -sideHalf ) xx += side ;
		 if ( yy < -sideHalf ) yy += side ;
		 if ( zz < -sideHalf ) zz += side ;
		 if ( xx >  sideHalf ) xx -= side ;
		 if ( yy >  sideHalf ) yy -= side ;
		 if ( zz >  sideHalf ) zz -= side ;

		 rd = xx*xx + yy*yy + zz*zz ;

		 if ( rd <= cutoffSquare) 
		 {
		    //#pragma omp atomic
		    int index = ninter++;
		    inter1 (index) = i;
		    inter2 (index) = j;

		    if ( ninter >= MAXINTERACT) perror("MAXINTERACT limit");
		 }
	     }
	   }
	   printf("ninter = %d, cutoff = %lf\n", ninter, cutoffRadius);

}

void InitCoordinates()
{
 int siz = boxSize, siz_3;
 int n, k,  j, i, npoints;

   printf("Init Coordinates ...\n");
   siz_3 = siz * siz * siz;
   npoints = siz_3 ; 
   for ( n =0; n< npoints; n++) {
      k   = n % siz ;
      j   = (int)((n-k)/siz) % siz;
      i   = (int)((n - k - j*siz)/(siz*siz)) % siz ; 

      x(n) = i*perturb ;
      y(n) = j*perturb ;
      z(n) = k*perturb ;

      x(n+npoints) = i*perturb + perturb * 0.5 ;
      y(n+npoints) = j*perturb + perturb * 0.5;
      z(n+npoints) = k*perturb ;

      x(n+npoints*2) = i*perturb + perturb * 0.5 ;
      y(n+npoints*2) = j*perturb ;
      z(n+npoints*2) = k*perturb + perturb * 0.5;

      x(n+npoints*3) = i*perturb ;
      y(n+npoints*3) = j*perturb + perturb * 0.5 ;
      z(n+npoints*3) = k*perturb + perturb * 0.5;
   }
 
} 

/*
!============================================================================
! Function  :  InitVelocities()
! Purpose   :
!    This routine initializes the velocities of the 
!    molecules according to a maxwellian distribution.
!============================================================================
*/

void  InitVelocities()
{
   int i, j, iseed, n_moles;
   double ekin, ts, sp, sc, r, s;
   double u1, u2, v1, v2, ujunk,tscale;
   double DRAND();

   printf("Init Velocities ...\n");
   n_moles = numMoles;
   iseed = 4711;
   ujunk = DRAND();  
   iseed = 0;
   tscale = (16.0)/(1.0*n_moles - 1.0);
   
   for ( j =0; j<3; j++) 
   {
   	for ( i =0; i< n_moles; i=i+2) 
   	{
   	  do 
	  {
   	    u1 = DRAND();
   	    u2 = DRAND();
   	    v1 = 2.0 * u1   - 1.0;
   	    v2 = 2.0 * u2   - 1.0;
   	    s  = v1*v1  + v2*v2 ;
   	  } while( s >= 1.0 );

   	  r = SQRT( -2.0*log(s)/s );

   	  if ( j == 0) 
	  {
   	    vhx(i)    = v1 * r;
   	    vhx(i+1)  = v2 * r;
   	  } 
	  else if ( j == 1) 
	  {
   	    vhy(i)    = v1 * r;
   	    vhy(i+1)  = v2 * r; 
   	  } 
	  else if ( j == 2) 
	  {
   	    vhz(i)    = v1 * r;
   	    vhz(i+1)  = v2 * r;
   	  }
   	}
   }       



/* There are three parts - repeat for each part */

   /*  Find the average speed in x direction */
   sp   = 0.0 ;
   for ( i=0; i<n_moles; i++) {
     sp = sp + vhx(i);
   } 
   sp   = sp/n_moles;


   /*  Subtract average from all velocities in x direction*/
   for ( i=0; i<n_moles; i++) {
     vhx(i) = vhx(i) - sp;
   }

   /*  Find the average speed for y direction */
   sp   = 0.0 ;
   for ( i=0; i<n_moles; i++) { 
     sp = sp + vhy(i);
   }
   sp   = sp/n_moles;

   /*  Subtract average from all velocities in y direction */
   for ( i=0; i<n_moles; i++) {
     vhy(i) = vhy(i) - sp;
   }

   /*  Find the average speed for z direction */
   sp   = 0.0 ;
   for ( i=0; i<n_moles; i++) { 
     sp = sp + vhz(i);
   }
   sp   = sp/(n_moles);

   /*  Subtract average from all velocities of 2nd part */
   for ( i=0; i<n_moles; i++) {
     vhz(i) = vhz(i) - sp;
   }

   /*  Determine total kinetic energy  */
   ekin = 0.0 ;
   for ( i=0 ; i< n_moles; i++ ) {
     ekin  = ekin  + vhx(i)*vhx(i) ; 
     ekin  = ekin  + vhy(i)*vhy(i) ;
     ekin  = ekin  + vhz(i)*vhz(i) ;
   }
   ts = tscale * ekin ;
   sc = timeStep * SQRT(TEMPERATURE/ts);
   for ( i=0; i< n_moles; i++) {
     vhx(i) = vhx(i) * sc ;
     vhy(i) = vhy(i) * sc ;
     vhz(i) = vhz(i) * sc ;
   }


}

/*
!============================================================================
!  Function :  InitForces()
!  Purpose :
!    Initialize all the partial forces to 0.0
!============================================================================
*/

void  InitForces()
{
int i, n_moles;

   printf("Init Forces ...\n");
   n_moles = numMoles;
   for ( i=0; i<n_moles; i++ ) {
     fx(i) = 0.0 ;
     fy(i) = 0.0 ;
     fz(i) = 0.0 ;
   } 
}


/* ----------- UTILITY ROUTINES & I/O ROUTINES ------- */

/*
!=============================================================
!  Function : drand_x()
!  Purpose  :
!    Used to calculate the distance between two molecules
!    given their coordinates.
!=============================================================
*/

double drand_x()
    {
      double  rand;
      static int nsd =1073741823;

      while (1) {
        nsd=   (16807*nsd) % 2147483647 ;
        if (nsd < 0) nsd = -nsd ;
        rand = (0.465661e-09)*(double)(nsd);
        if ( rand > 0.0  &&  rand < 1.0 ) return rand;
        if (nsd == 0)  {
          nsd = 1073741823;
          printf("\n  ERROR in RAND - SEED RESET TO DEFAULT");
        }
      }
    }

/*
!=============================================================
!  Function : PrintResults()
!  Purpose  :
!    Prints out the final accumulated results 
!=============================================================
*/
void PrintResults(int move, double ekin, double vel, double count)
{
 double ek, etot, temp, pres, rp, tscale ;

   ek   = 24.0 * ekin ;
   epot = 4.00 * epot ;
   etot = ek + epot ;
   tscale = (16.0)/((double)numMoles - 1.0);
   temp = tscale * ekin ;
   pres = DENSITY * 16.0 * (ekin-vir)/numMoles ;
   vel  = vel/numMoles;
   rp   = (count/(double)(numMoles)) * 100.0 ;

   printf("\n------------------------------------------- \n");
   printf("Results  \n");
   printf(
     "\n %4d %12.4f %12.4f %12.4f %8.4f %8.4f %8.4f %5.1f",
        move, ek,    epot,   etot,   temp,   pres,   vel,     rp);
   printf("\n\n In the final step there were %d interacting pairs\n", ninter);
}

#ifdef BLOCK_ADAPTIVE

#define xyz2n(x, y, z) ((z) * (xd * yd) + (y) * xd + (x)) 
     
void
find_neighbor_blk(int x, int y, int z, int xd, int yd, int zd, int *neiblk) 
{
    int i, j, k, n = 0; 
    int x1[3], y1[3], z1[3];

    x1[0] = (x == 0)? xd-1 : x-1;    
    x1[1] = x;
    x1[2] = (x == xd-1)? 0 : x+1;    
    y1[0] = (y == 0)? yd-1 : y-1;    
    y1[1] = y;
    y1[2] = (y == yd-1)? 0 : y+1;    
    z1[0] = (z == 0)? zd-1 : z-1;    
    z1[1] = z;
    z1[2] = (z == zd-1)? 0 : z+1;    

    for (i=0; i<3; i++)
    for (j=0; j<3; j++)
    for (k=0; k<3; k++) {
        neiblk[n++] = xyz2n (x1[i], y1[j], z1[k]);    
    }
}

void
RebuildInteraction(double (*xyz)[3], int n_nodes, 
		   double cutoffRadius, double side)
{
    int *nodes, *start, n_xyz[3], neiblk[27], xd,yd,zd, x,y,z;
    int i, j, k, l, m, ii, jj, n_blk;
    double sideHalf, cutoffSquare, rd, xx, yy, zz;

    sideHalf = side * 0.5;
    cutoffSquare = (cutoffRadius * TOLERANCE)*(cutoffRadius * TOLERANCE);
    ninter = 0;

    nodes = NodeBlocking(xyz, n_nodes, (cutoffRadius * TOLERANCE), n_xyz);
    start = nodes + n_nodes;

    xd = n_xyz[0]; yd = n_xyz[1]; zd = n_xyz[2];
    n_blk = xd * yd * zd;

    if (n_blk < 27) {
	/* too few blocks cause potentially wrong result */
	printf("Too few blocks!\n");
	exit(0);
    }

#ifdef CHECK
{
    int *check, check_OK=1;
 
    MALLOC(check, int, n_nodes)
    bzero((void*)check, sizeof(int)*n_nodes);
    for (i=0; i<n_blk; i++)
    for (j=start[i]; j<start[i+1]; j++) {
	k = nodes[j];
	if (check[k] == 1) printf("doubly checked (%d)\n", k);
	else check[k] = 1;
    }

    for (i=0; i<n_nodes; i++)
        if (check[i] == 0) { 
	   printf("not covered (%d)\n", i);
	   check_OK = 0;
	}
    FREE(check, int, n_nodes)
    if (check_OK) printf("check is OK \n");
}
#endif CHECK

    for (l = 0; l < n_blk; l++) {
        x = l % xd;
        y = (l / xd) % yd;
        z = l / (xd * yd);

        find_neighbor_blk(x, y, z, xd, yd, zd, neiblk); 

        for (k = 0; k < 27; k++) {

	    m = neiblk[k];

            for (ii = start[l]; ii < start[l+1]; ii++) {
              for (jj = start[m]; jj < start[m+1]; jj++) {

                 i = nodes[ii];
	         j = nodes[jj];

                 if (j <= i) continue;  /* no distinction in order (i,j) */

		 xx = x(i) - x(j);
		 yy = y(i) - y(j);
		 zz = z(i) - z(j);

		 if ( xx < -sideHalf ) xx += side ;
		 if ( yy < -sideHalf ) yy += side ;
		 if ( zz < -sideHalf ) zz += side ;
		 if ( xx >  sideHalf ) xx -= side ;
		 if ( yy >  sideHalf ) yy -= side ;
		 if ( zz >  sideHalf ) zz -= side ;

		 rd = xx*xx + yy*yy + zz*zz ;

		 if ( rd <= cutoffSquare) {
		    inter1 (ninter) = i;
		    inter2 (ninter) = j;
		    ninter ++;

		    if ( ninter >= MAXINTERACT) perror("MAXINTERACT limit");
		 }
	    }}
        }
    } 
    printf("ninter = %d, cutoff = %lf\n", ninter, cutoffRadius);
    sort_edgelist (ninter, n_nodes, inter);

    FREE (nodes, int, 2*n_blk+1)
}
#endif

