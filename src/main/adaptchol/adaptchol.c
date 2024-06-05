
/* 
 
 ==========================================================================
 
 ADAPTCHOL.C -- a program for doing MCMC with multivariate adaptions.
 
 Copyright (c) 2004, 2006 by Gareth O. Roberts and Jeffrey S. Rosenthal
 
 2024: Minimal fixes to get working and satisfy standards compliance by Louis Aslett
 
 Licensed for general copying, distribution and modification according to
 the GNU General Public License (http://www.gnu.org/copyleft/gpl.html).
 
 ----------------------------------------------------
 
 Save as "adaptchol.c".
 
 Compile with "gcc adaptchol.c -o adaptchol -lm", then run with "./adaptchol".
 
 Upon completion, can run 'source("adaptx")' in R to see a trace plot
 of the first coordinate.  Also, can run '<<adaptmath' in Mathematica to
 analyse the speed-ups according to the eigenvalues.
 
 The parameters at the top (NUMITS, DIM, etc.) may be modified (then recompile).
 
 ==========================================================================
 
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

/* #define NUMITS 1000000 */
#define NUMITS 10000

/* #define DIM 200 */
#define DIM 200

#define beta 0.05  /* set beta = 0.0 for purest adapting */

#define initstepmultiplier 2

#define identitymultiplier 0.1
/* #define identitymultiplier 2.38 */

#define PRINTSPACING 1000

/* #define bspacing (NUMITS / 20) */

#define MATHFILE "adaptmath"
#define XFILE "adaptx"
#define RFILE "adaptR"
#define EFILE "adapte"
#define BFILE "adaptb"
#define AFILE "adapta"

#define PI 3.14159265

double targetinvcov[DIM][DIM];

double targlogdens(double w[DIM]);
int seedrand(void);
double normal(void);
int imin(int a, int b);

int main(int argc, char *argv[])
  
{
  
  int i,j,k,t, numaccept, xspacing, bspacing, aspacing;
  double X[DIM], Y[DIM];
  double empcov[DIM][DIM], invtargetsqrt[DIM][DIM];
  double chol[DIM][DIM], normalsvec[DIM];
  double meanx[DIM], oldmeanx[DIM];
  double covsumsq[DIM][DIM], prevcovsumsq[DIM][DIM], compmat[DIM][DIM];
  double ell, A, tmpsum;
  FILE *fpmath, *fpx, *fpr, *fpe, *fpb, *fpa;
  struct timeval currtv;
  double begintime, endtime;
  /* long beginsec, beginusec, endsec, endusec; */
  
  /* INITIALISATIONS. */
  seedrand();
  ell = 2.381204;
  for (i=0; i<DIM; i++) {
    X[i] = meanx[i] = 0.0;
    for (j=0; j<DIM; j++) {
      /* invtargetsqrt[i][j] = normal(); */
      if (i==j) {
        invtargetsqrt[i][j] = /* 10 + */ normal();
      } else {
        invtargetsqrt[i][j] = normal();
      }
      covsumsq[i][j] = chol[i][j] = 0.0;
    }
  }
  numaccept = 0;
  if ((fpx = fopen(XFILE,"w")) == NULL) {
    fprintf(stderr, "Unable to write to file %s.\n", XFILE);
  }
  if ((fpa = fopen(AFILE,"w")) == NULL) {
    fprintf(stderr, "Unable to write to file %s.\n", AFILE);
  }
  if ((fpe = fopen(EFILE,"w")) == NULL) {
    fprintf(stderr, "Unable to write to file %s.\n", EFILE);
  }
  
  if ((fpb = fopen(BFILE,"w")) == NULL) {
    fprintf(stderr, "Unable to write to file %s.\n", BFILE);
  }
  fprintf(fpb, "\nRrat<-function(x)\n");
  fprintf(fpb, "{\n");
  fprintf(fpb, "eigs<-eigen(x)$values\n");
  fprintf(fpb, "sum(eigs^(-2))*length(eigs)/(sum(eigs^(-1))^2)\n");
  fprintf(fpb, "}\n\n");
  fprintf(fpb, "bvector = NULL\n\n");
  fprintf(fpx, "\nxvector <- c(");
  fprintf(fpa, "\navector <- c(");
  
  if ((fpmath = fopen(MATHFILE,"w")) == NULL) {
    fprintf(stderr, "Unable to write to file %s.\n", MATHFILE);
  }
  if ((fpr = fopen(RFILE,"w")) == NULL) {
    fprintf(stderr, "Unable to write to file %s.\n", RFILE);
  }
  
  bspacing = imin(10000, NUMITS / 20);
  
  xspacing = NUMITS / 10000;
  if (xspacing == 0)
    xspacing = 1;
  
  aspacing = DIM;  /* Like xspacing, but smaller to compute acf etc. */
      
      /* COMPUTE RANDOM TARGET INVERSE COVARIANCE MATRIX. */
      for (i=0; i<DIM; i++) {
        for (j=0; j<DIM; j++) {
          targetinvcov[i][j] = 0.0;
          for (k=0; k<DIM; k++) {
            targetinvcov[i][j] = targetinvcov[i][j] + 
              invtargetsqrt[i][k] * invtargetsqrt[j][k];
          }
        }
      }
      
      /* OUTPUT IT. */
      fprintf(fpb, "siginv = matrix( c(");
      for (i=0; i<DIM; i++) {
        for (j=0; j<DIM; j++) {
          fprintf(fpb, " %f", targetinvcov[i][j]);
          if ( (i<DIM-1) || (j<DIM-1) ) {
            fprintf(fpb, ",");
          }
        }
      }
      fprintf(fpb, "), nrow=%d)\n\n", DIM);
      fprintf(fpb, "sig = solve(siginv)\n\n");
      
      /* Output initial efficiency information. */
      fprintf(fpmath, "A = { ");
      fprintf(fpr, "\n\nA = matrix( c( ");
      for (i=0; i<DIM; i++) {
        fprintf(fpmath, "{ ");
        for (j=0; j<DIM; j++) {
          fprintf(fpmath, " %f", targetinvcov[i][j]);
          fprintf(fpr, "%f", targetinvcov[i][j]);
          if (j < DIM-1) {
            fprintf(fpmath, ", ");
            fprintf(fpr, ", ");
          }
        }
        fprintf(fpmath, "}");
        if (i < DIM-1) {
          fprintf(fpmath, ", \n");
          fprintf(fpr, ", ");
        }
      }
      fprintf(fpmath, "}; \n");
      fprintf(fpr, "), ncol=%d ) \n\n", DIM);
      fprintf(fpmath, "DIM = %d; \n", DIM);
      fprintf(fpmath, "eigensys = Eigensystem[A]; \n");
      fprintf(fpmath, "L1 = Sum[ 1.0/eigensys[[1,i]], {i,1,DIM} ]; \n");
      fprintf(fpmath, "ss = Sum[ 1.0/(eigensys[[1,i]]^2), {i,1,DIM} ]; \n");
      fprintf(fpmath, "initratio = DIM * ss / L1^2; \n");
      fprintf(fpmath, "Print[\" \"] \n");
      fprintf(fpmath, "Print[\"initial ratio = \", initratio] \n\n");
      
      /* fflush(fpmath);
       fflush(fpr); */
      fflush(NULL);
      
      /* MAIN ITERATIVE LOOP. */
      gettimeofday(&currtv, (struct timezone *)NULL);
      begintime = 1.0*currtv.tv_sec + 0.000001*currtv.tv_usec;
      printf("\n\nBeginning, dimension=%d, time=%f.\n\n",
             DIM, begintime);
      for (t=1; t<=NUMITS; t++) {
        
        if (t == PRINTSPACING * (t/PRINTSPACING)) {
          fflush(fpx);
          fflush(fpa);
          printf("Iteration %d of %d ... \n", t, NUMITS);
        }
        
        /* OBTAIN VECTOR OF STANDARD NORMALS. */
        for (i=0; i<DIM; i++)
          normalsvec[i] = normal();
        
        if ( (t <= initstepmultiplier*DIM) || (drand48() < beta) ) {
          
          /* JUST USE MULTIPLE OF IDENTITY FOR PROPOSAL COVARIANCE. */
          
          for (i=0; i<DIM; i++) {
            for (j=0; j<DIM; j++) {
              Y[i] = X[i] + identitymultiplier * normalsvec[i] / sqrt(DIM);
            }
          }
        } else {
          
          /* USE EMPIRICAL COVARIANCE AS PROPOSAL COVARIANCE. */
          
          for (i=0; i<DIM; i++) {
            for (j=0; j<DIM; j++) {
              empcov[i][j] = covsumsq[i][j] / (t-DIM) ;
            }
          }
          
          /* COMPUTE CHOLESKY DECOMPOSITION OF EMPCOV. */
          
          /* First compute chol[j][0]: */
          chol[0][0] = sqrt( empcov[0][0] );
          for (j=1; j<DIM; j++) {
            chol[j][0] = empcov[j][0] / chol[0][0];
          }
          
          /* Then compute rest. */
          for (i=1; i<DIM; i++) {
            tmpsum = 0.0;
            for (k=0; k<i; k++) {
              tmpsum = tmpsum + chol[i][k]*chol[i][k];
            }
            chol[i][i] = sqrt( empcov[i][i] - tmpsum );
            for (j=i+1; j<DIM; j++) {
              tmpsum = 0.0;
              for (k=0; k<i; k++) {
                tmpsum = tmpsum + chol[j][k]*chol[i][k];
              }
              chol[j][i] = ( empcov[j][i] - tmpsum ) / chol[i][i];
            }
          }
          
          /* GENERATE PROPOSAL VALUE. */
          for (i=0; i<DIM; i++) {
            Y[i] = X[i];
            for (j=0; j<=i; j++) {
              Y[i] = Y[i] + (ell / sqrt(DIM)) * chol[i][j]*normalsvec[j];
            }
          }
          
        } /* End of "if" statement. */
          
          /* COMPUTE ACCEPT/REJECT VALUE. */
          A = targlogdens(Y) - targlogdens(X);
        if ( log(drand48()) < A) {
          /* Accept the proposal. */
          for (i=0; i<DIM; i++) {
            X[i] = Y[i];
          }
          numaccept++;
        }
        
        if (t == xspacing * (t/xspacing)) {
          /* Write X[0] to file. */
          if (t > xspacing) /* Not first one printed, so need comma. */
          fprintf(fpx, ",");
          fprintf(fpx, " %f", X[0]);
        }
        
        if (t == aspacing * (t/aspacing)) {
          /* Write X[0] to file. */
          if (t > aspacing) /* Not first one printed, so need comma. */
          fprintf(fpa, ",");
          fprintf(fpa, " %f", X[0]);
        }
        
        /* Update meanx[i]. */
        for (i=0; i<DIM; i++) {
          oldmeanx[i] = meanx[i];
          meanx[i] = ((1.0*t)/(t+1))*oldmeanx[i] + (1.0/(t+1))*X[i];
        }
        
        /* UPDATE EMPIRICAL COVARIANCE. */
        
        /* Store previous empirical covariance matrix. */
        for (i=0; i<DIM; i++) {
          for (j=0; j<DIM; j++) {
            prevcovsumsq[i][j] = covsumsq[i][j];
          }
        }
        
        /* Compute new one. */
        for (i=0; i<DIM; i++) {
          for (j=0; j<DIM; j++) {
            covsumsq[i][j] = prevcovsumsq[i][j] + 
              (t-1)*(oldmeanx[i]-meanx[i])*(oldmeanx[j]-meanx[j]) +
              (X[i]-meanx[i])*(X[j]-meanx[j]) ;
          }
        }
        
        /* Compute and output matrices for BFILE. */
        if (t == bspacing * (t/bspacing)) {
          
          /* COMPUTE COMPMAT MATRIX. */
          for (i=0; i<DIM; i++) {
            for (j=0; j<DIM; j++) {
              compmat[i][j] = 0.0;
              for (k=0; k<DIM; k++) {
                compmat[i][j] = compmat[i][j] + targetinvcov[i][k] * empcov[k][j];
              }
            }
          }
          
          /* OUTPUT IT. */
          fprintf(fpb, "\nB%d = matrix( c( ", t/bspacing);
          for (i=0; i<DIM; i++) {
            for (j=0; j<DIM; j++) {
              fprintf(fpb, "%f", compmat[i][j]);
              if (j < DIM-1) {
                fprintf(fpb, ", ");
              }
            }
            if (i < DIM-1) {
              fprintf(fpb, ", ");
            }
          }
          fprintf(fpb, "), ncol=%d ) \n\n", DIM);
          fprintf(fpb, "bvector = c(bvector, Rrat(B%d))\n\n", t/bspacing);
          fflush(fpb);
          
        }
        
      } /* End of "t" loop. */
          
          gettimeofday(&currtv, (struct timezone *)NULL);
      endtime = 1.0*currtv.tv_sec + 0.000001*currtv.tv_usec;
      printf("\n\nEnding, dimension=%d, time=%f, ellapsed=%f secs.\n\n",
             DIM, endtime, endtime-begintime );
      
      /* COMPUTE COMPMAT MATRIX. */
      for (i=0; i<DIM; i++) {
        for (j=0; j<DIM; j++) {
          compmat[i][j] = 0.0;
          for (k=0; k<DIM; k++) {
            compmat[i][j] = compmat[i][j] + targetinvcov[i][k] * empcov[k][j];
          }
        }
      }
      
      /* OUTPUT SOME VALUES TO EFILE. */
      
      printf("\nDone!  Details in files %s, %s, %s, %s, and %s.\n\n",
             EFILE, XFILE, RFILE, BFILE, MATHFILE);
      
      fprintf(fpe, "\nDone ... final sumsq matrix is:\n\n");
      for (i=0; i<DIM; i++) {
        for (j=0; j<DIM; j++) {
          fprintf(fpe, "  %f", covsumsq[i][j]);
        }
        fprintf(fpe, "\n");
      }
      
      fprintf(fpe, "\nDone ... final meanx vector is:\n\n");
      for (i=0; i<DIM; i++) {
        fprintf(fpe, "  %f", meanx[i]);
      }
      
      fprintf(fpe, "\n\nDone ... invtargetsqrt matrix was:\n\n");
      for (i=0; i<DIM; i++) {
        for (j=0; j<DIM; j++) {
          fprintf(fpe, "  %f", invtargetsqrt[i][j]);
        }
        fprintf(fpe, "\n");
      }
      
      fprintf(fpe, "\n\nDone ... targetinvcov matrix was:\n\n");
      for (i=0; i<DIM; i++) {
        for (j=0; j<DIM; j++) {
          fprintf(fpe, "  %f", targetinvcov[i][j]);
        }
        fprintf(fpe, "\n");
      }
      
      fprintf(fpe, "\n\nDone ... final empirical covariance matrix is:\n\n");
      for (i=0; i<DIM; i++) {
        for (j=0; j<DIM; j++) {
          fprintf(fpe, "  %f", empcov[i][j]);
        }
        fprintf(fpe, "\n");
      }
      
      fprintf(fpe, "\n\nDone ... cholesky matrix L is:\n\n");
      for (i=0; i<DIM; i++) {
        for (j=0; j<DIM; j++) {
          fprintf(fpe, "  %f", chol[i][j]);
        }
        fprintf(fpe, "\n");
      }
      fprintf(fpe, "\n");
      
      fprintf(fpe, "\n\nDone ... L L^T matrix is:\n\n");
      for (i=0; i<DIM; i++) {
        for (j=0; j<DIM; j++) {
          tmpsum = 0.0;
          for (k=0; k<DIM; k++)
            tmpsum = tmpsum + chol[i][k] * chol[j][k];
          fprintf(fpe, "  %f", tmpsum);
        }
        fprintf(fpe, "\n");
      }
      fprintf(fpe, "\n");
      
      fprintf(fpe, "\n\nDone ... final compmat matrix is:\n\n");
      for (i=0; i<DIM; i++) {
        for (j=0; j<DIM; j++) {
          fprintf(fpe, "  %f", compmat[i][j]);
        }
        fprintf(fpe, "\n");
      }
      
      fprintf(fpe, "\nDIM = %d;  NUMITS = %d;  Acceptance rate = %f \n", 
              DIM, NUMITS, ((float)numaccept) / ((float)NUMITS) );
      
      fprintf(fpe, "\n\n\n");
      
      /* PREPARE OUTPUT FOR MATHEMATICA AND R ... */
      
      /* OLD: [Note: Could instead use R, with "eigen()$values", "solve()"
       for inverse, "t()" for transpose, "chol()", etc.] */
      
      /* Can now just do 'source("adaptR")' in R ... */
      
      /* Compute final efficiency. */
      fprintf(fpmath, "B = { ");
      fprintf(fpr, "\nB = matrix( c( ");
      for (i=0; i<DIM; i++) {
        fprintf(fpmath, "{ ");
        for (j=0; j<DIM; j++) {
          fprintf(fpmath, " %f", compmat[i][j]);
          fprintf(fpr, "%f", compmat[i][j]);
          if (j < DIM-1) {
            fprintf(fpmath, ", ");
            fprintf(fpr, ", ");
          }
        }
        fprintf(fpmath, "}");
        if (i < DIM-1) {
          fprintf(fpmath, ", \n");
          fprintf(fpr, ", ");
        }
      }
      fprintf(fpmath, "}; \n");
      fprintf(fpmath, "DIM = %d; \n", DIM);
      fprintf(fpmath, "eigensys = Eigensystem[B]; \n");
      fprintf(fpmath, "L1 = Sum[ 1.0/eigensys[[1,i]], {i,1,DIM} ]; \n");
      fprintf(fpmath, "ss = Sum[ 1.0/(eigensys[[1,i]]^2), {i,1,DIM} ]; \n");
      fprintf(fpmath, "finalratio = DIM * ss / L1^2; \n");
      fprintf(fpmath, "Print[\" \"] \n");
      fprintf(fpmath, "Print[\"final ratio = \", finalratio] \n\n");
      fprintf(fpmath, "Print[\" \"] \n");
      
      fprintf(fpr, "), ncol=%d ) \n\n\n", DIM);
      fprintf(fpr, "Rrat<-function(x)\n");
      fprintf(fpr, "{\n");
      fprintf(fpr, "eigs<-eigen(x)$values\n");
      fprintf(fpr, "sum(eigs^(-2))*length(eigs)/(sum(eigs^(-1))^2)\n");
      fprintf(fpr, "}\n");
      fprintf(fpr, "print(\"Initial R =\")\n");
      fprintf(fpr, "print(Rrat(A))\n");
      fprintf(fpr, "print(\"Final R=\")\n");
      fprintf(fpr, "print(Rrat(B))\n");
      fprintf(fpr, "\n");
      
      /* Exit the program cleanly. */
      /* fprintf(fpx, "\177"); -- No, don't need this. */
      fprintf(fpx, " )\n");
      fprintf(fpx, "\nplot(xvector, type='l')\n\n");
      fprintf(fpa, " )\n");
      fprintf(fpa, "\nplot(avector, type='l')\n\n");
      fclose(fpmath);
      fclose(fpx);
      fclose(fpa);
      fclose(fpr);
      fclose(fpe);
      fprintf(fpb, "print(bvector)\n\n\n");
      fclose(fpb);
      return(0);
      
}


/* TARGLOGDENS */
double targlogdens( double w[DIM] )
{
  int ii, jj;
  double tmpval;
  
  tmpval = 0.0;
  for (ii=0; ii<DIM; ii++) {
    for (jj=0; jj<DIM; jj++) {
      tmpval = tmpval - 0.5 * w[ii]*w[jj]*targetinvcov[ii][jj];
    }
  }
  
  return(tmpval);
}


/* SEEDRAND: SEED RANDOM NUMBER GENERATOR. */
int seedrand(void)
{
  int seed;
  struct timeval tmptv;
  gettimeofday (&tmptv, (struct timezone *)NULL);
  /* seed = (int) (tmptv.tv_usec - 1000000 *
   (int) ( ((double)tmptv.tv_usec) / 1000000.0 ) ); */
  seed = (int) tmptv.tv_usec;
  srand48(seed);
  (void)drand48();  /* Spin it once. */
  return(0);
}


/* NORMAL:  return a standard normal random number. */
double normal(void)
{
  double R, theta;
  
  R = - log(drand48());
  theta = 2 * PI * drand48();
  
  return( sqrt(2*R) * cos(theta));
}


int imin(int a, int b)
{
  if (a < b)
    return(a);
  else
    return(b);
}
