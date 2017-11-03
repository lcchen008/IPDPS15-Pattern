/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National 
   Laboratories ( http://software.sandia.gov/mantevo/ ). The primary 
   authors of miniMD are Steve Plimpton and Paul Crozier 
   (pscrozi@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you 
   can redistribute it and/or modify it under the terms of the GNU Lesser 
   General Public License as published by the Free Software Foundation; 
   either version 3 of the License, or (at your option) any later 
   version.
  
   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
   Lesser General Public License for more details.
    
   You should have received a copy of the GNU Lesser General Public 
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov). 

   Please read the accompanying README and LICENSE files.
---------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

#include "ljs.h"
#include "atom.h"
#include "force.h"
#include "neighbor.h"
//#include "integrate.h"
#include "thermo.h"
//#include "comm.h"
#include "timer.h"
#include "irregular/irregular_runtime.h"
#include "regular/regular_runtime.h"
#include "include/runtime.h"

#define MAXLINE 256
#define NPROC 32 

int input(In &, Atom &, Force &, Neighbor &, int &ntimes, double &dt, Thermo &);
void create_box(Atom &, int, int, int, double);
int create_atoms(Atom &, int, int, int, double);
void create_velocity(double, Atom &, Thermo &);

int main(int argc, char **argv)
{
  int me;
  //MPI_Init(&argc,&argv);
  //MPI_Comm_rank(MPI_COMM_WORLD,&me);
  RuntimeInit(argc, argv);

  In in;
  Atom atom;
  Force force;
  Neighbor neighbor;
  //Integrate integrate;
  Thermo thermo;
  int ntimes;
  double dt;

  int flag;
  //while (1) {
    flag = input(in,atom,force,neighbor,ntimes,dt,thermo);

    create_box(atom,in.nx,in.ny,in.nz,in.rho);

    flag = neighbor.setup(atom);

    force.setup();

    thermo.setup(in.rho,ntimes);

    printf("nx: %d, ny: %d, nz: %d, rho: %f\n", in.nx, in.ny, in.nz, in.rho);

    flag = create_atoms(atom,in.nx,in.ny,in.nz,in.rho);

    printf("atoms created...\n");

    create_velocity(in.t_request,atom,thermo);

    printf("velocity created...\n");

    neighbor.build(atom);

    printf("build done...\n");

    //create edges based on the neighbor list
    EDGE *edges = (EDGE *)malloc(sizeof(EDGE)*neighbor.total_neigh);

    printf("edges created...\n");

    printf("edges: %d, natoms: %d\n", neighbor.total_neigh, atom.nlocal);

    int count = 0;
    for(int i = 0; i < atom.nlocal; i++)
    {
   	int *neighs = neighbor.firstneigh[i];	 
	int numneigh = neighbor.numneigh[i];

	for(int k = 0; k < numneigh; k++)
	{
		int j = neighs[k];	
		edges[count].idx0 = i;
		edges[count].idx1 = j;
		count++;
	}
    }

    //create node data
    double *nodes = (double *)malloc(sizeof(double)*3*atom.nlocal);
    memcpy(nodes, &(atom.x)[0][0], sizeof(double)*3*atom.nlocal); 

    double cutforcesq = force.cutforcesq;

    printf("nodes created...\n");

    //MPI_Barrier(MPI_COMM_WORLD);

    //create runtime
    IrregularRuntime runtime
    (
   	neighbor.total_neigh,
	atom.nlocal,
	edges,
	NULL,
	nodes,
	0,
	sizeof(double)*3,
	sizeof(double)*3,
	3,
	nodes,
	sizeof(double),
	&cutforcesq,
	sizeof(double),
	NPROC,
	1
    );
    runtime.IrregularInit();

    //compute thermo
    Offset *offsets= (Offset *)malloc(sizeof(Offset)*neighbor.total_neigh);

    double *nodes1 = (double *)malloc(sizeof(double)*3*atom.nlocal + sizeof(double));
    *nodes1 = cutforcesq;
    double *p = nodes1 + 1;
    memcpy(p, nodes, sizeof(double)*3*atom.nlocal);

    for(int i = 0; i<neighbor.total_neigh; i++)
    {
   	offsets[i].offset = i * sizeof(int)*2; 
	offsets[i].size = sizeof(int)*2;
    }

    RegularRuntime gr(NPROC, edges, sizeof(EDGE)*neighbor.total_neigh, offsets, neighbor.total_neigh, nodes, sizeof(double)*3*atom.nlocal);

    gr.RegularInit();

    MPI_Barrier(MPI_COMM_WORLD);


for(int iter = 0; iter < 5; iter++)
{
    printf("==========>ITER %d\n", iter);
    runtime.IrregularStart();
    //gr.RegularStart();
}

    //thermo.compute(0,atom,neighbor,force);

    //integrate.run(atom,force,neighbor,comm,thermo,timer);

    //thermo.compute(-1,atom,neighbor,force);

    //output(in,atom,force,neighbor,comm,thermo,integrate,timer);
  //}
}
