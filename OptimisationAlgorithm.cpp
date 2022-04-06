
#include <petsctao.h>
#include <math.h>
#include <iostream>
#include <petscmat.h>
#include <petscvec.h>
#include <petscsnes.h>
#include <petscpc.h>
#include <petsctao.h>
#include <complex>
#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <vector>
#include <list>
#include <fftw3.h>
#include <string.h>
 #include <unistd.h>
static  char help[]="";
using namespace std;
#define INTERACTIVE

typedef  std::complex<PetscScalar> mycomplex;

/*typedef struct {
  PetscInt n; /* Length x */
  //PetscInt ne; /* number of equality constraints */
  //PetscInt ni; /* number of inequality constraints */
/*  Vec      x,xl,xu;
  Vec      ce,ci,bl,bu;
  Mat      Ae,Ai,H;
  Vec      S;
} AppCtx;*/


//first only sequential
struct ConfigureProgramRun {
	int filetype;  //1 =gnuplot 2 = rohan
	std::list<std::string>  inputfiles;
	std::string lineshapefilename;
	std::string dosfilename;
	std::string lineshapesolutionfilename;
	std::string dossolutionfilename;
	std::string outputdir;
	int areaextend;
	int skipfactor;
	PetscReal intereststart;
	PetscReal interestend;
	PetscReal absintereststart;
	PetscReal absinterestend;

	int applyDOS; // 0 no 1 yes
	PetscReal applyDOSfactor; // defaults to 1. !

	std::string outfilesuffix;
	int initialcond;// 1 random 2 constant 3 guess
	mycomplex initialvalue;

	int initialcondrelax;// 1 random 2 constant 3 guess
	PetscScalar initialvaluerelax;
	mycomplex phasefactor;

	int initialconddos;// 1 random 2 constant 3 guess
	PetscScalar initialvaluedos;


	std::vector<struct ConfigureObjective> confobj;
//	std::vector<struct ConfigureObjectiveRelax> confobjrel;
	//std::vector<struct ConfigureObjectiveDOS> confobjdos;

};


struct ConfigureObjective {
	PetscReal norm_factor;//=10.*0.01*0.1*0.1;
	PetscReal norm_imag;//=0.;//5;
	PetscReal norm_diff;//=0.0*0.125;//5; // 20K
	PetscReal norm_factor_sq;//=10.*0;
	PetscReal norm_factor_ct;
	PetscReal norm_err;//=1.;

	//new stuff
	PetscReal norm_diff_normed;
	PetscReal diff_normed_para;

	// accuracy solver
	PetscReal gatol,grtol;

};

struct workcontext {
  Vec X;
//  ierr = VecCreateSeq(PETSC_COMM_SELF,wctx.nmax_s*wctx.nmax_s,&X);CHKERRQ(ierr);
/*  PetscReal norm_factor;//=10.*0.01*0.1*0.1;
	PetscReal norm_imag;//=0.;//5;
	PetscReal norm_diff;//=0.0*0.125;//5; // 20K
	PetscReal norm_factor_sq;//=10.*0;
	PetscReal norm_factor_ct;
	PetscReal norm_err;//=1.;
  */
	unsigned int nmax_s;
	unsigned int nmiddle_s; //marks 0 at the sfunc
	unsigned int nmax_l;
	unsigned int nmiddle_l; //marks 0 at the lfunc
	bool matinit;
  Vec S;
	Vec spec; //global vector!
  //cout<<"The number entered by user is "<<nmax_s<< endl;
  //cout<<"The number entered by user is "<<nmax_l<< endl;
	Vec speclin; //global vector absorption spectrum

	unsigned int nabs_start; //marks the beginning of the interesting part of the absorption spectrum
	unsigned int nabs_end; //marks the beginning of the interesting part of the absorption spectrum


	//Vec xcur;

	PetscInt localsize;
	/*PetscInt localsize_dos;
	PetscInt localsize_relax;*/
	PetscInt ncpus;

	PetscReal discr;
	PetscReal scaleadjust;
	PetscReal startspec;

	PetscReal scaleadjustlin;

	mycomplex phasefactor;



	// benchmark

	Vec tempvec;
	Vec diffvec;
	Vec rdiffvec;

	// benchmark linear

	Vec tempveclin;
	Vec diffveclin;
	Vec rdiffveclin;


	PetscScalar resabs;
	PetscScalar resimag;
	PetscScalar reserr;
	PetscScalar resdiff;

	PetscScalar traceerr;
	PetscScalar resdiffdiag;
	PetscScalar resdiffodiag;
	PetscScalar ressqnorm;

	PetscInt objnum;
	PetscReal gatol= 0.1;
  PetscReal grtol=0.1;
  PetscReal gttol=0.1;
/*
	struct ConfigureObjective confobj;
	/*struct ConfigureObjectiveRelax confobj_relax;
	struct ConfigureObjectiveDOS confobj_dos;
	PetscInt objnum;

	Vec lvecext; // for the greenfunction case, for extract run
	Vec lvecrel; // runs on the relaxation case, can be the same as lvecext

	Vec dossqrt; //sqrt of density of states
	// Hessian
	Vec hessvec;*/

	std::string outfilesuffix;
};


//typedef  PetscScalar mycomplex;
#define CMPL_SIZE_FAC 2
PetscErrorCode InitializeProblem(workcontext *);
PetscErrorCode Bounds(Tao tao,Vec low,Vec up,void*user)
{
	struct workcontext *wctx=(struct workcontext*)user;
	PetscReal *lowarr;
	PetscReal *uparr;
	VecGetArray(low,( PetscScalar**) &lowarr);
	VecGetArray(up,( PetscScalar**) &uparr);

	int nmax_l=wctx->nmax_l;
	int nmiddle_l=wctx->nmiddle_l;
#if 0
	for(int i=0; i<length;i++) {
		lowarr[2*i]=PETSC_NINFINITY;
		lowarr[2*i+1]=PETSC_NINFINITY;
		uparr[2*i]=PETSC_INFINITY;
		uparr[2*i+1]=0;
	}
#else
	for (int m1=0; m1<nmax_l;m1++) {
		int m2=0;
		for (; m2<nmiddle_l;m2++) {
			lowarr[2*((m2)*nmax_l+m1)]=0; //Re positive
			lowarr[2*((m2)*nmax_l+m1)+1]=PETSC_NINFINITY;//Im negative
			uparr[2*((m2)*nmax_l+m1)]=PETSC_INFINITY; // Re positive
			uparr[2*((m2)*nmax_l+m1)+1]=0; //Im negative
		}
		//m2=nmiddle
		lowarr[2*((m2)*nmax_l+m1)]=PETSC_NINFINITY; //Re full range
		lowarr[2*((m2)*nmax_l+m1)+1]=PETSC_NINFINITY;//Im negative
		uparr[2*((m2)*nmax_l+m1)]=PETSC_INFINITY; // Re full range
		uparr[2*((m2)*nmax_l+m1)+1]=0; //Im negative



		for (m2=nmiddle_l+1;m2<nmax_l;m2++) {
			lowarr[2*((m2)*nmax_l+m1)]=PETSC_NINFINITY; //Re negative
			lowarr[2*((m2)*nmax_l+m1)+1]=PETSC_NINFINITY;//Im negative
			uparr[2*((m2)*nmax_l+m1)]=0; // Re negative
			uparr[2*((m2)*nmax_l+m1)+1]=0; //Im negative

		}

	}
#endif

	VecRestoreArray(low,( PetscScalar**) &lowarr);
	VecRestoreArray(up,( PetscScalar**) &uparr);

}

PetscErrorCode DestroyProblem(workcontext *);
PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal *,Vec,void *);
/*PetscErrorCode FormHessian(Tao,Vec,Mat,Mat, void*);
PetscErrorCode FormInequalityConstraints(Tao,Vec,Vec,void*);
PetscErrorCode FormEqualityConstraints(Tao,Vec,Vec,void*);
PetscErrorCode FormInequalityJacobian(Tao,Vec,Mat,Mat, void*);
PetscErrorCode FormEqualityJacobian(Tao,Vec,Mat,Mat, void*);*/


PetscErrorCode InitializeProblem(workcontext *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*user->n = 2;
  ierr = VecCreateSeq(PETSC_COMM_SELF,user->n,&user->x);CHKERRQ(ierr);
  ierr = VecDuplicate(user->x,&user->xl);CHKERRQ(ierr);
  ierr = VecDuplicate(user->x,&user->xu);CHKERRQ(ierr);
  ierr = VecSet(user->x,0.0);CHKERRQ(ierr);
  ierr = VecSet(user->xl,-1.0);CHKERRQ(ierr);
  ierr = VecSet(user->xu,2.0);CHKERRQ(ierr);

  user->ne = 1;
  ierr = VecCreateSeq(PETSC_COMM_SELF,user->ne,&user->ce);CHKERRQ(ierr);

  user->ni = 2;
  ierr = VecCreateSeq(PETSC_COMM_SELF,user->ni,&user->ci);CHKERRQ(ierr);*/
  VecCreate(PETSC_COMM_SELF,&user->S); //self will create a local vector
  VecSetType(user->S,VECSEQ);

  VecSetSizes(user->S,PETSC_DECIDE,user->nmax_s*user->nmax_s*CMPL_SIZE_FAC);

  mycomplex *s;
    ierr = VecGetArray(user->S,(PetscScalar**) &s);CHKERRQ(ierr);
  //Wo muss ich S speichern, ist *s=Stemp richtig?
  printf("%x s %d %d\n",s,ierr, user->nmax_s);

  /*VecCreate(PETSC_COMM_SELF,&user->S); //self will create a local vector
  VecSetType(user->S,VECSEQ);
  VecSetSizes(user->S,PETSC_DECIDE,user->nmax_s*user->nmax_s*CMPL_SIZE_FAC);*/

/*  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,user->ne,user->n,user->n,NULL,&user->Ae);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,user->ni,user->n,user->n,NULL,&user->Ai);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Ae);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Ai);CHKERRQ(ierr);


  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,user->n,user->n,1,NULL,&user->H);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->H);CHKERRQ(ierr);*/

  PetscFunctionReturn(0);
}

/*PetscErrorCode DestroyProblem(AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&user->Ae);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Ai);CHKERRQ(ierr);
  ierr = MatDestroy(&user->H);CHKERRQ(ierr);

  ierr = VecDestroy(&user->x);CHKERRQ(ierr);
  ierr = VecDestroy(&user->ce);CHKERRQ(ierr);
  ierr = VecDestroy(&user->ci);CHKERRQ(ierr);
  ierr = VecDestroy(&user->xl);CHKERRQ(ierr);
  ierr = VecDestroy(&user->xu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}*/
//initialize before reading the Spectrum


// gnuplot style data !
int read2DspectrumIn(const char *filename,struct workcontext * wctx, int skip_factor, int append_number )
{
	FILE *file=fopen(filename,"r");
	if (!file) return 0;

	/*gkogkas
	char *line;
	line=NULL;
	size_t len;
	int l=0;
	while(getline(&line, &len, filename)!=-1;){l++;
	const char s[2]=" ";
		char *token;
		 // get the first token
   token = strtok(line, s);}*/
	//auto tuning algorithm
	unsigned int spec_points1=0;
	unsigned int spec_points2=0;
	PetscReal spec_disc1=0;
	PetscReal spec_disc2=0;
	PetscReal spec_start1=0;
	PetscReal spec_start2=0;



	std::vector<mycomplex> spec_data;
	unsigned int spec_size=1000;
	spec_data.resize(spec_size);

	char *readbuffer=(char*)malloc(1000);
	unsigned long int rbufsize=1000;



	for (unsigned int i1=0;i1<spec_points1 || spec_points1==0;i1++) {
		if (spec_points2!=0) spec_data.resize(spec_points2*(i1+1));
		PetscReal om1=spec_disc1*((PetscReal)i1)+spec_start1;
		for (unsigned int i2=0;i2<spec_points2 || spec_points2==0;i2++) {
			printf("spec_disc2 %d\n",i2);
			PetscReal om2=spec_disc2*((PetscReal)i2)+spec_start2;
			ssize_t readbytes=getline(&readbuffer,&rbufsize,file);



			float omin1,omin2,valre,valim,valabs;
			int num=0;
			if (readbytes>0) num=sscanf(readbuffer,"%g %g %g %g %g\n",&omin1,&omin2,&valre,&valim,&valabs);
			if (num!=5 || readbytes<0) {
				if (spec_points2==0) { // we have detected the maximum size
					spec_points2=i2;
					PetscPrintf(PETSC_COMM_WORLD,"Spectrum points2 %d detected \n",spec_points2);
					break;
				} else if (readbytes<0 && i2==0) break;
				else{
					PetscPrintf(PETSC_COMM_WORLD,"File is broken! num!=5 abort \n"); exit(1);
				}
			}
			if (i1==0) {
				spec_start1=omin1;
				om1=omin1;
			} else if (i1==1 && spec_disc1==0.) {
				if (spec_disc1==0) spec_disc1=omin1-spec_start1;
				om1=omin1;
				PetscPrintf(PETSC_COMM_WORLD,"Detected axis1 scale %g %g\n",spec_start1,spec_disc1);
			}
			if (i2==0) {
				spec_start2=omin2;
				om2=omin2;
			} else if (i2==1 && spec_disc2==0.) {
				spec_disc2=omin2-spec_start2;
				om2=omin2;
				PetscPrintf(PETSC_COMM_WORLD,"Detected axis2 scale %g %g\n",spec_start2,spec_disc2);
			}
			if ((fabs(om1-omin1)> spec_disc1*0.01 && spec_disc1>0.)
					|| (fabs(om2-omin2)> spec_disc2*0.01 && spec_disc2>0.)) {
				PetscPrintf(PETSC_COMM_WORLD,"Axis do not match incompatible scale! Abort\n");
				PetscPrintf(PETSC_COMM_WORLD,"%g %g %g %g\n",omin1,omin2,om1,om2);
				exit(1);
			}
			if (i1==0 && i2>=spec_size) {
				spec_data.resize(2*spec_size);
				spec_size=2*spec_size;

			}
			spec_data[i1*spec_points2+i2]=wctx->phasefactor*mycomplex(valre,valim);
		}
		if (i1!=0) getline(&readbuffer,&rbufsize,file);
		//fscanf(file,"\n");
		if (feof(file)) {
			spec_points1=i1;
			PetscPrintf(PETSC_COMM_WORLD,"Spectrum points1 %d detected \n",spec_points1);
		}
	}
	fclose(file);
	free(readbuffer);

	// now we have the data in storage
	// we have to reformat the data for the algorithm
	// first we have to note that one axis has to be inverted
	// and everything the diagonal has to be the diagonal in the array structure!
	if (spec_disc2!=spec_disc1) {
		PetscPrintf(PETSC_COMM_WORLD,"Discretizations do not match! Abort\n");
		exit(-1);
	}
	// we calculate now the area, which we need to store!
	PetscReal spec_min=std::min(-(spec_start1+spec_disc1*spec_points1),spec_start2);
	PetscReal spec_max=std::max((spec_start2+spec_disc2*spec_points2),-spec_start1);
	wctx->discr=spec_disc1;
	wctx->startspec=spec_min;
	//wctx->discr2=spec_disc2;
	int upscale=1; //mechanism to expand the area
	int nmax_s=wctx->nmax_s=wctx->nmax_s=(int)((spec_max-spec_min)/spec_disc1)/skip_factor*upscale;

	wctx->nmiddle_s=wctx->nmax_s/2+wctx->nmax_s %2;
	//now initialize the bigger computational area
	int nmax_l=wctx->nmax_l=wctx->nmax_s+append_number;
	int nmiddle_l=wctx->nmiddle_l=wctx->nmiddle_s+append_number/2+append_number%2;

	//we can now allocate the vector spectrum vector // this is done on every cpu!
	VecCreate(PETSC_COMM_SELF,&wctx->spec); //self will create a local vector
	VecSetType(wctx->spec,VECSEQ);

	VecSetSizes(wctx->spec,PETSC_DECIDE,wctx->nmax_s*wctx->nmax_s*CMPL_SIZE_FAC);
	//VecDuplicate(wctx->xcur,&wctx->spec);

	VecZeroEntries(wctx->spec);

	mycomplex * svec;
	VecGetArray(wctx->spec,(PetscScalar**) &svec);


	int upscaleshift=(upscale-1)*nmax_s/upscale/upscale;
	int oldnmax=nmax_s/upscale;
	for (int n1=0;n1<oldnmax;n1++) {
		int i1=-(int)((spec_min+spec_start1)/spec_disc1+0.5)-n1*skip_factor;
		if (i1<0 || i1>=oldnmax) {
			PetscPrintf(PETSC_COMM_WORLD,"Invalid i1 %d %d %d",i1,n1,nmax_s);
			continue;
		}
		for (int n2=0;n2<oldnmax;n2++) {
			int i2=(int)((spec_min-spec_start2)/spec_disc2+0.5)+n2*skip_factor;
			if (i2<0 || i2>=oldnmax) {
				PetscPrintf(PETSC_COMM_WORLD,"Invalid i2 %d %d %d %g %g",i2,n2,nmax_s,spec_min,spec_start2);
				continue;
			}
			svec[(n1+upscaleshift)*nmax_s+(n2+upscaleshift)]=spec_data[i1*spec_points2+i2];


		}
	}
printf("svec%g\n",svec[0] );
	VecRestoreArray(wctx->spec,(PetscScalar**) &svec);

	if (wctx->scaleadjust>0) {
		VecScale(wctx->spec,1./wctx->scaleadjust);
	} else {
		PetscReal specnorm;
		VecNorm(wctx->spec,NORM_2,&specnorm);
		//specnorm*=10.;
		wctx->scaleadjust=specnorm;
	VecScale(wctx->spec,1./specnorm);
printf("1/specnorm %g\n", 1./specnorm);	}

  cout<<"The number nmax_s is "<<nmax_s<< endl;
  cout<<"The number nmax_l is "<<nmax_l<< endl;
  cout<<"The number nmiddle_l is "<<nmiddle_l<< endl;
	return 1;
}
/*macro für die Funktionswerte, die vom gespeicherten Vektor x abgelesen werden sollen*/
inline int xc(int e,int n,int m, int nmax){
return e+2*n+2*nmax*m;}
inline int yc(int e,int n,int m, int nmax, int mmax){
return 2*nmax*mmax+e+2*n+2*nmax*m;}
inline int nc(int n1,int n2, int nmax){
return n1+n2*nmax;}

void printspec(struct workcontext *wctx){
  mycomplex * svec;
  VecGetArray(wctx->spec,(PetscScalar**) &svec);
  for(int n2=0; n2<wctx->nmax_s ; n2++){
  for(int n1=0; n1<wctx->nmax_s ; n1++){
    mycomplex Stemp=-svec[nc(n1,n2,wctx->nmax_s)];
    printf("stemp %g\n", Stemp);}};
  VecRestoreArray(wctx->spec,(PetscScalar**) &svec);

}

PetscErrorCode FormFunctionGradient(Tao tao, Vec X,  PetscReal *f, Vec G, void *user)
{
  mycomplex       *g;
  const mycomplex *x;
  PetscErrorCode    ierr;
  	struct workcontext *wctx=(struct workcontext*)user;
    //ConfigureObjective &confobj=wctx->confobj;
    int nmax_s= wctx->nmax_s;
    //int nmiddle_s=wctx->nmiddle_s;
    int nmax_l= wctx->nmax_l;
    //int nmiddle_l=wctx->nmiddle_l;

    PetscReal norm_factor=10;//=10.*0.01*0.1*0.1;
    PetscReal norm_imag=0.;//5;
    PetscReal norm_diff=0.0*0.125;//5; // 20K
    PetscReal norm_factor_sq=10;//=10.*0;
    PetscReal norm_factor_ct=10;
    PetscReal norm_err=1.;
  //Wie kann ich das ConfigureObjective benutzen ohne hier nochmal alles zu  schreiben?

//VecZeroEntries(wctx->spec);
  //PetscFunctionBegin;
  ierr = VecGetArrayRead(X,(const PetscScalar**) &x);CHKERRQ(ierr);
  ierr = VecGetArray(G,(PetscScalar**)&g);CHKERRQ(ierr);
  mycomplex *svec;
  ierr = VecGetArrayRead(wctx->spec,(const PetscScalar**)&svec);CHKERRQ(ierr);

mycomplex *s;
  ierr = VecGetArray(wctx->S,(PetscScalar**) &s);CHKERRQ(ierr);
//Wo muss ich S speichern, ist *s=Stemp richtig?
printf("%x s %d %x\n",s,ierr, svec);
mycomplex ftemp=0;
      for(int n2=0; n2<nmax_s ; n2++){
		  for(int n1=0; n1<nmax_s ; n1++){
        mycomplex Stemp=-svec[nc(n1,n2,nmax_s)];

        mycomplex temp2=0;
        s[nc(n1,n2,nmax_s)]=Stemp;
			  for(int m=0; m<nmax_l ; m++){
				  for(int e=0; e<2 ; e++){
					  for(int ep=0; ep<2 ; ep++){
	Stemp+=conj(x[xc(e,n1,m,nmax_s)])*(x[xc(ep,n2,m,nmax_s)]-x[yc(e,n2,m,nmax_s,nmax_l)]);
  temp2+=conj(x[(xc(e,n1,n1,nmax_s))])*x[(xc(e,n1,n1,nmax_s))]+conj(x[(yc(e,n1,n1,nmax_s,nmax_l))])*x[(yc(e,n1,n1,nmax_s,nmax_l))];
}}}
ftemp+=conj(Stemp)*Stemp
        +(norm_factor*n1/*Betrag nicht nötig?*/+norm_factor_sq*n1*n1+norm_factor_ct)*temp2;
 s[nc(n1,n2,nmax_s)]=Stemp;
}}printf("ftemp-- %g\n", ftemp.real());
*f=ftemp.real();


PetscReal ReX=0;
PetscReal ImX=0;
PetscReal ReY=0;
PetscReal ImY=0;
  for(int nn=0; nn<nmax_s ; nn++){
  for(int mm=0; mm<nmax_s ; mm++){
    for(int ee=0; ee<2 ; ee++){
      g[xc(ee,nn,mm,nmax_s)]=0;
      for(int n=0; n<2 ; n++){
        for(int e=0; e<2 ; e++){
          ReX=s[nc(nn,n,nmax_s)].real()*(x[xc(e,n,mm,nmax_l)]+x[yc(ee,n,mm,nmax_l,nmax_l)]).real()+s[nc(n,nn,nmax_s)].real()*x[xc(e,n,mm,nmax_l)].real()+s[nc(n,nn,nmax_s)].imag()*(x[xc(e,n,mm,nmax_l)]+x[yc(ee,n,mm,nmax_l,nmax_l)]).imag()-s[nc(n,nn,nmax_s)].imag()*x[xc(e,n,mm,nmax_l)].imag()
              +2*x[xc(ee,nn,mm,nmax_l)].real();
          ImX=-s[nc(nn,n,nmax_s)].real()*(x[xc(e,n,mm,nmax_l)]+x[yc(ee,n,mm,nmax_l,nmax_l)]).imag()-s[nc(n,nn,nmax_s)].real()*x[xc(e,n,mm,nmax_l)].imag()-s[nc(n,nn,nmax_s)].imag()*(x[xc(e,n,mm,nmax_l)]+x[yc(ee,n,mm,nmax_l,nmax_l)]).imag()-s[nc(n,nn,nmax_s)].imag()*x[xc(e,n,mm,nmax_l)].real()
              +2*x[xc(ee,nn,mm,nmax_l)].imag();
          ReY=-s[nc(nn,n,nmax_s)].imag()*x[yc(ee,n,mm,nmax_l,nmax_s)].imag()
              +2*x[yc(ee,nn,mm,nmax_l,nmax_s)].real();
          ImY=s[nc(nn,n,nmax_s)].imag()*x[yc(ee,n,mm,nmax_l,nmax_s)].real()
              +2*x[yc(ee,nn,mm,nmax_l,nmax_s)].real();
          g[xc(ee,nn,mm,nmax_s)]+=mycomplex(ReX,ImX);
          g[yc(ee,nn,mm,nmax_s,nmax_l)]+=mycomplex(ReY,ImY);
        }
      }
    }
  }
  }

  ierr = VecRestoreArrayRead(wctx->spec,(const PetscScalar**) &svec);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,(const PetscScalar**) &x);CHKERRQ(ierr);
  ierr = VecRestoreArray(G,(PetscScalar**)&g);CHKERRQ(ierr);
  ierr = VecRestoreArray(wctx->S,(PetscScalar**)&s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
PetscErrorCode FormInequalityConstraints(Tao tao, Vec X, Vec CI, void *ctx)
{
  const PetscScalar *x;
  PetscScalar       *c;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(CI,&c);CHKERRQ(ierr);
  c[0] = x[0]*x[0] - x[1];
  c[1] = -x[0]*x[0] + x[1] + 1.0;
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(CI,&c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormEqualityConstraints(Tao tao, Vec X, Vec CE,void *ctx)
{
  PetscScalar    *x,*c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(CE,&c);CHKERRQ(ierr);
  c[0] = x[0]*x[0] + x[1] - 2.0;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(CE,&c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormInequalityJacobian(Tao tao, Vec X, Mat JI, Mat JIpre,  void *ctx)
{
  PetscInt          rows[2];
  PetscInt          cols[2];
  PetscScalar       vals[4];
  const PetscScalar *x;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  rows[0] = 0;       rows[1] = 1;
  cols[0] = 0;       cols[1] = 1;
  vals[0] = +2*x[0]; vals[1] = -1.0;
  vals[2] = -2*x[0]; vals[3] = +1.0;
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = MatSetValues(JI,2,rows,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(JI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode FormEqualityJacobian(Tao tao, Vec X, Mat JE, Mat JEpre, void *ctx)
{
  PetscInt          rows[2];
  PetscScalar       vals[2];
  const PetscScalar *x;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  rows[0] = 0;       rows[1] = 1;
  vals[0] = 2*x[0];  vals[1] = 1.0;
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = MatSetValues(JE,1,rows,2,rows,vals,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(JE,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JE,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
*/

/*TEST

   build:
      requires: !complex !define(PETSC_USE_CXX)

   test:
      requires: superlu
      args: -tao_smonitor -tao_view -tao_gatol 1.e-5

TEST*/
/*void saveSolution(const char* outfile,const char*vectorfile, struct workcontext * wctx)
{
	FILE * file=fopen(outfile,"w");
	PetscSynchronizedFPrintf(PETSC_COMM_WORLD,file, "ObjectTag SpectrumInfo\n");
	PetscSynchronizedFPrintf(PETSC_COMM_WORLD,file, "StartSpectrum %g\n",wctx->startspec);
	PetscSynchronizedFPrintf(PETSC_COMM_WORLD,file, "Scalefactor %g\n",wctx->scaleadjust);
	PetscSynchronizedFPrintf(PETSC_COMM_WORLD,file, "Discretization %g\n",wctx->discr);
	PetscSynchronizedFPrintf(PETSC_COMM_WORLD,file, "NumberPointsSmall %d\n",wctx->nmax_s);
	PetscSynchronizedFPrintf(PETSC_COMM_WORLD,file, "MiddlePointSmall %d\n",wctx->nmiddle_s);
	PetscSynchronizedFPrintf(PETSC_COMM_WORLD,file, "NumberPointsLarge %d\n",wctx->nmax_l);
	PetscSynchronizedFPrintf(PETSC_COMM_WORLD,file, "MiddlePointLarge %d\n",wctx->nmiddle_l);
	PetscSynchronizedFPrintf(PETSC_COMM_WORLD,file, "Lineshape %s/%s.bin\n",dirname((char*)outfile),vectorfile);


	PetscSynchronizedFPrintf(PETSC_COMM_WORLD,file, "ObjectTag SpectrumInfo End\n");
	fclose(file);
	char sname[strlen(vectorfile)+1+4];
	sprintf(sname,"%s/%s.bin",dirname((char*)outfile),vectorfile);
	PetscViewer viewer;
	PetscViewerBinaryOpen(PETSC_COMM_WORLD,sname, FILE_MODE_WRITE ,&viewer);
	VecView(wctx->lvecext,viewer);
	PetscViewerDestroy(&viewer);
}


bool readSolution(const char *infile, struct workcontext *wctx)
{
	FILE * file=fopen(infile,"r");
	char name[2048];
	double temp;
	if (fscanf(file, "ObjectTag SpectrumInfo\n")!=0) return false;
	if (fscanf(file, "StartSpectrum %lg\n\n",&temp)!=1) return false;
	wctx->startspec=temp;
	if (fscanf(file, "Scalefactor %lg\n\n",&temp)!=1) return false;
	wctx->scaleadjust=temp;
	if (fscanf(file, "Discretization %lg\n\n",&temp)!=1) return false;
	wctx->discr=temp;
	PetscPrintf(PETSC_COMM_WORLD,"Solution is read in, set discr to %g", wctx->discr);
	int tempi=0;
	if (fscanf(file, "NumberPointsSmall %d\n\n",&tempi)!=1) return false;
	wctx->nmax_s=tempi;
	if (fscanf(file, "MiddlePointSmall %d\n\n",&tempi)!=1) return false;
	wctx->nmiddle_s=tempi;
	if (fscanf(file, "NumberPointsLarge %d\n\n",&tempi)!=1) return false;
	wctx->nmax_l=tempi;
	if (fscanf(file, "MiddlePointLarge %d\n\n",&tempi)!=1) return false;
	wctx->nmiddle_l=tempi;
	if (fscanf(file, "Lineshape %s\n",name)!=1) return false;
	if (fscanf(file, "ObjectTag SpectrumInfo End\n")!=0) return false;

	std::string vector_name=std::string(name);

	PetscViewer viewer;
	Vec vec;
	VecCreate(PETSC_COMM_WORLD,&vec);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD,vector_name.c_str(), FILE_MODE_READ ,&viewer);
	VecLoad(vec,viewer);

	wctx->lvecext=vec;
	PetscViewerDestroy(&viewer);

	fclose(file);
	return true;
}*/
void writeDebugDataOut2D(const char *filename,const mycomplex *data, PetscInt size1, PetscInt size2, struct workcontext* wctx )
{
	int n1,n2;
	FILE *file=fopen(filename,"w");
	for (n1=0;n1<size1;n1++) {
		PetscReal om1=wctx->startspec+((PetscReal)n1)*wctx->discr;
		for (n2=0;n2<size2;n2++) {
			mycomplex val=data[n1*size2+n2];
			PetscReal om2=wctx->startspec+((PetscReal)n2)*wctx->discr;
			fprintf(file,"%d %d %g %g %g %g %g\n",n1,n2,om1,om2,std::real(val),std::imag(val),std::abs(val));
		}
		fprintf(file,"\n");
	}
	fclose(file);
}

/*PetscErrorCode  KSPMyMonitor(KSP ksp,PetscInt its,PetscReal fgnorm,void *ctx)
{

	struct workcontext *wctx=(struct workcontext*)ctx;
	KSPLSQRMonitorDefault(ksp,its,fgnorm,PETSC_NULL);
	Vec resid;
	KSPBuildResidual(ksp,NULL,NULL,&resid);
	if (its<20 || its%40==0) {
		PetscScalar *resarr;
		VecGetArray(resid,&resarr);
		writeDebugDataOut2D("debugresid.txt",(mycomplex*)resarr, wctx->nmax_s, wctx->nmax_s,wctx);

		VecRestoreArray(resid,&resarr);
		VecAXPBY(resid,1.,-1.,wctx->spec);
		VecGetArray(resid,&resarr);
		writeDebugDataOut2D("debugcurspec.txt",(mycomplex*)resarr, wctx->nmax_s, wctx->nmax_s ,wctx);

		VecRestoreArray(resid,&resarr);

		KSPBuildSolution(ksp,resid,NULL);

		VecGetArray(resid,&resarr);
		writeDebugDataOut2D("debugsol.txt",(mycomplex*)resarr, wctx->nmax_s, wctx->nmax_s,wctx );

		VecRestoreArray(resid,&resarr);

		VecDestroy(&resid);
	}

}
*/
PetscErrorCode TaoMyMonitor(Tao tao, void *ctx)
{
   PetscInt       its;
   PetscReal      fct,gnorm;
   PetscViewer    viewer;


   struct workcontext *wctx=(struct workcontext*)ctx;

   int nmax_s_1= wctx->nmax_s;
   int nmax_s_2= wctx->nmax_s;
   int nmax_l= wctx->nmax_l;


   TaoConvergedReason reason;



   TaoGetSolutionStatus(tao, &its, &fct, &gnorm, NULL, NULL, &reason);


   PetscPrintf(PETSC_COMM_WORLD,"a:%2.3g i: %2.3g e: %2.4g d: %2.3g", wctx->resabs,wctx->resimag,
		   wctx->reserr,wctx->resdiff);
   PetscPrintf(PETSC_COMM_WORLD,"iter = %3D,",its);
   PetscPrintf(PETSC_COMM_WORLD," Function value: %g,",(double)fct);
   if (gnorm >= PETSC_INFINITY) {
	   PetscPrintf(PETSC_COMM_WORLD,"  Residual: Inf \n");
   } else {
	   PetscPrintf(PETSC_COMM_WORLD,"  Residual: %g \n",(double)gnorm);
   }
   mycomplex *svec;
   VecGetArrayRead(wctx->spec,(const PetscScalar**) &svec);

   //mycomplex *tvec;
   //VecGetArrayRead(wctx->tempvec,(const PetscScalar**) &tvec);

   //mycomplex *dvec;
   //VecGetArrayRead(wctx->diffvec,(const PetscScalar**) &dvec);

   //mycomplex *rvec;
   //VecGetArrayRead(wctx->rdiffvec,(const PetscScalar**) &rvec);

   Vec G;
     TaoGetGradientVector(tao, &G);
   mycomplex *gvec;
   VecGetArrayRead(G,(const PetscScalar**) &gvec);

   Vec X;
   TaoGetSolutionVector(tao,&X);
   const mycomplex *xvec;
   VecGetArrayRead(X,(const PetscScalar**) &xvec);
#ifdef INTERACTIVE
   writeDebugDataOut2D("/tmp/debugspecl.txt",xvec, nmax_l, nmax_l ,wctx);
   writeDebugDataOut2D("/tmp/debugspecs.txt",svec, nmax_s_1, nmax_s_2 ,wctx);
   //writeDebugDataOut2D("/tmp/debugspect.txt",tvec, nmax_s_1, nmax_s_2 ,wctx);
   //writeDebugDataOut2D("/tmp/debugspecd.txt",dvec, nmax_s_1, nmax_s_2 ,wctx);
   //writeDebugDataOut2D("/tmp/debugspecr.txt",rvec, nmax_s_1, nmax_s_2 ,wctx);
   writeDebugDataOut2D("/tmp/debugspecg.txt",gvec, nmax_l, nmax_l ,wctx);
#else
   if ((its) % 20 == 0 || reason!=TAO_CONTINUE_ITERATING ) {
	   char buffer[100];
	   sprintf(buffer,"%d",wctx->objnum);
	   std::string objnum(buffer);


	   writeDebugDataOut2D(("resultspecl"+wctx->outfilesuffix+"_"+objnum+".txt").c_str(),xvec, nmax_l, nmax_l ,wctx);
	   writeDebugDataOut2D(("resultspecs"+wctx->outfilesuffix+"_"+objnum+".txt").c_str(),svec, nmax_s_1, nmax_s_2 ,wctx);
	   //writeDebugDataOut2D(("resultspect"+wctx->outfilesuffix+"_"+objnum+".txt").c_str(),tvec, nmax_s_1, nmax_s_2 ,wctx);
	   //writeDebugDataOut2D(("resultspecd"+wctx->outfilesuffix+"_"+objnum+".txt").c_str(),dvec, nmax_s_1, nmax_s_2 ,wctx);
	   //writeDebugDataOut2D(("resultspecr"+wctx->outfilesuffix+"_"+objnum+".txt").c_str(),rvec, nmax_s_1, nmax_s_2 ,wctx);
	   writeDebugDataOut2D(("resultspecg"+wctx->outfilesuffix+"_"+objnum+".txt").c_str(),gvec, nmax_l, nmax_l ,wctx);

   }


#endif


   VecRestoreArrayRead(wctx->spec,(const PetscScalar**) &svec);
   VecRestoreArrayRead(X,(const PetscScalar**) &xvec);
   VecRestoreArrayRead(G,(const PetscScalar**) &gvec);

   //VecRestoreArrayRead(wctx->tempvec,(const PetscScalar**) &tvec);
   //VecRestoreArrayRead(wctx->diffvec,(const PetscScalar**) &dvec);
   //VecRestoreArrayRead(wctx->rdiffvec,(const PetscScalar**) &rvec);


   return(0);
}



PetscErrorCode main(int argc,char **argv)
{
struct ConfigureProgramRun cpr;
  PetscErrorCode     ierr;                /* used to check for functions returning nonzeros */
  Tao                tao;
  struct workcontext wctx;  //heißt es dass wctx auf die Werte vom workcontext zeigt?
;
wctx.phasefactor=1.;

  //AppCtx             user;                /* application context */
 if(argc!=2){printf("wrong usage\n" );exit(-1);}
  ierr = PetscInitialize(&argc,&argv,(char *)0,help);if (ierr) return ierr;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n---- Optimisation algorithm -----\n");CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Start extract properties\n");


read2DspectrumIn("filename.dat", &wctx, 1 , 0.1);

ierr = InitializeProblem(&wctx);CHKERRQ(ierr);//wctx0make
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);//02419e-320
  ierr = TaoSetType(tao,TAOBLMVM);CHKERRQ(ierr);//wctx0
  Vec X;  //wctx0
  ierr = VecCreateSeq(PETSC_COMM_SELF,2*2*2*wctx.nmax_s*wctx.nmax_s,&X);CHKERRQ(ierr);//wctx0
  PetscReal norm;VecSetRandom(X,NULL);VecNorm(X,NORM_2,&norm);VecScale(X,1./norm); //wctx2.122e-314
  ierr = TaoSetInitialVector(tao,X);CHKERRQ(ierr); // wctx2.02419e-320
  //ierr = TaoSetVariableBounds(tao,user.xl,user.xu);CHKERRQ(ierr);
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void*)&wctx);//wctx5.9842e-318
CHKERRQ(ierr);

  /*		TaoSetVariableBoundsRoutine(tao, Bounds,(void*)&wctx);The number entered by user
  		TaoSetVariableBoundsRoutine(tao, DOSBounds,(void*)&wctx);
  		TaoSetObjectiveAndGradientRoutine(tao, FormFunctionGradientRelax , &wctx);
  		TaoSetVariableBoundsRoutine(tao, RelaxBounds,(void*)&wctx);
*/
   /*ierr = TaoSetEqualityConstraintsRoutine(tao,user.ce,FormEqualityConstraints,(void*)&user);CHKERRQ(ierr);
  ierr = TaoSetInequalityConstraintsRoutine(tao,user.ci,FormInequalityConstraints,(void*)&user);CHKERRQ(ierr);

 ierr = TaoSetJacobianEqualityRoutine(tao,user.Ae,user.Ae,FormEqualityJacobian,(void*)&user);CHKERRQ(ierr);
  ierr = TaoSetJacobianInequalityRoutine(tao,user.Ai,user.Ai,FormInequalityJacobian,(void*)&user);CHKERRQ(ierr);
  ierr = TaoSetJacobianInequalityRoutine(tao,user.Ai,user.Ai,FormInequalityJacobian,(void*)&user);CHKERRQ(ierr);
*/
  /* ierr = TaoSetTolerances(tao,0,0,0);CHKERRQ(ierr); */

  /*
      This algorithm produces matrices with zeros along the diagonal therefore we need to use
    SuperLU which does partial pivoting
  */

  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);//wctx5.9842e-318
  TaoSetMonitor(tao,TaoMyMonitor,(void *)&wctx,NULL);//wctx0
  		/*for (int cobjs=0;cobjs<cpr.confobj.size();cobjs++) {
  			wctx.confobj=cpr.confobj[cobjs];
  			wctx.objnum=cobjs;
  			PetscPrintf(PETSC_COMM_WORLD,"Start iteration for %d objective\n",cobjs);
  			PetscReal gatol,grtol,gttol;
  			TaoSetTolerances(tao, wctx.confobj.gatol,wctx.confobj.grtol,PETSC_DEFAULT);
  			TaoGetTolerances(tao,  &gatol, &grtol, &gttol);

  			PetscPrintf(PETSC_COMM_WORLD,"\nTolerances  gatol %g grtol %g gttol %g\n",gatol,grtol,gttol);
  			PetscPrintf(PETSC_COMM_WORLD,"\nParameters %g %g %g %g %g %g\n",wctx.confobj.norm_diff,wctx.confobj.norm_err,
  					wctx.confobj.norm_factor,wctx.confobj.norm_factor_sq,wctx.confobj.norm_factor_ct, wctx.confobj.norm_imag);

*/

PetscReal gatol=0.01;
PetscReal grtol=0.01;
PetscReal gttol=0.01;
TaoSetTolerances(tao, wctx.gatol,wctx.grtol,PETSC_DEFAULT);
TaoGetTolerances(tao,  &gatol, &grtol, &gttol);//printspec(&wctx);
  			TaoSolve(tao); //wctx0
  			TaoConvergedReason reason;
  			TaoGetConvergedReason(tao,&reason);//wctx0


  			PetscPrintf(PETSC_COMM_WORLD, "\nreason %d\n", reason);
  		//}



  		TaoDestroy(&tao);
  		//wctx.lvecext=xresu;


  //ierr = TaoSolve(tao);CHKERRQ(ierr);


  //ierr = DestroyProblem(&wctx);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
