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
#include "mpi.h"

#include "ljs.h"
#include "atom.h"
#include "force.h"
#include "neighbor.h"
//#include "integrate.h"
#include "thermo.h"

#define MAXLINE 256

int input(In &in, Atom &atom, Force &force,
	  Neighbor &neighbor, int &ntimes, double &dt, Thermo &thermo)
{
  FILE *fp;
  int flag;
  char line[MAXLINE];

    fp = fopen("lj.in","r");
    if (fp == NULL) flag = 0;
    else flag = 1;

  
    fgets(line,MAXLINE,fp);
    fgets(line,MAXLINE,fp);
    fgets(line,MAXLINE,fp);
    sscanf(line,"%d %d %d",&in.nx,&in.ny,&in.nz);
    fgets(line,MAXLINE,fp);
    sscanf(line,"%d",&ntimes);
    fgets(line,MAXLINE,fp);
    sscanf(line,"%d %d %d",&neighbor.nbinx,&neighbor.nbiny,&neighbor.nbinz);
    fgets(line,MAXLINE,fp);
    sscanf(line,"%lg",&dt);
    fgets(line,MAXLINE,fp);
    sscanf(line,"%lg",&in.t_request);
    fgets(line,MAXLINE,fp);
    sscanf(line,"%lg",&in.rho);
    fgets(line,MAXLINE,fp);
    sscanf(line,"%d",&neighbor.every);
    fgets(line,MAXLINE,fp);
    sscanf(line,"%lg %lg",&force.cutforce,&neighbor.cutneigh);
    fgets(line,MAXLINE,fp);
    sscanf(line,"%d",&thermo.nstat);
    fclose(fp);

  return 0;
}
