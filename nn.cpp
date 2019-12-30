// neural network logic
#include "chess.h"
#include <xmmintrin.h>
#include <math.h>
#include "nn-weights.h"
#include "threads.h"
#include "vectorutils.h"

// ts entry data structure
typedef struct{
	short int score;				// deep evaluation score.
	unsigned char piece[32];		// cell values are: 0=empty, 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king. Plus 8 if black(top bit). So, 0-14. Use 1/2 byte.
	unsigned char c1:1,c2:1,c3:1,c4:1,player:1,remarks:3;
		// 4: castling possible: c1=white lower(Q), c2=white upper(K), c3=black lower(q), c4=black upper(k). 1=allowed, 0=not allowed.
		// 1: player. 0/1 for w/b.
		// 3: remarks: 0 is unsolved. 1 is fruit_231 for 3 sec. 2 is sf5 for 3 sec.
	unsigned char last_move;		// last move made to, for ep captures only
	unsigned char fullmoveclock;	// can be up to 100 or more - use byte for 0-255
	unsigned char dummy[3];
} ts_entry; // 2+32+1*6=40 bytes.

// ts entry2 data structure
typedef struct{
	short int score_deep;				// deep evaluation score.
	unsigned char piece[32];		// cell values are: 0=empty, 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king. Plus 8 if black(top bit). So, 0-14. Use 1/2 byte.
	unsigned char c1:1,c2:1,c3:1,c4:1,player:1,remarks:3;
		// 4: castling possible: c1=white lower(Q), c2=white upper(K), c3=black lower(q), c4=black upper(k). 1=allowed, 0=not allowed.
		// 1: player. 0/1 for w/b.
		// 3: remarks: 0 is unsolved. 1 is fruit_231 for 3 sec. 2 is sf5 for 3 sec.
	unsigned char last_move;		// last move made to, for ep captures only
	short int score_shallow;
	unsigned char dummy[2];
} ts_entry2; // 2+32+1*6=40 bytes.

typedef struct {
	double mgw;
	board b;
	short int score_deep;
	short int score_shallow;
	unsigned char fullmoveclock;
} board_plus;

#if ENGINE
void convert_TS_to_board(board *b, ts_entry *ts){
	unsigned int i;
	static const unsigned char r[]={0,65,66,67,68,69,70,0,0,129,130,131,132,133,134};
	//								0  1  2  3  4  5  6 7 8   9  10  11  12  13  14

	// first, piece[] values
	for(i=0;i<64;i+=2){
		unsigned char v=ts->piece[i/2];
		b->piece[i]=r[v&15];
		b->piece[i+1]=r[v>>4];
	}

	// second, castling
	b->castle=ts->c1;
	b->castle+=2*ts->c2;
	b->castle+=4*ts->c3;
	b->castle+=8*ts->c4;

	// third, player
	b->player=ts->player+1; // turn 0/1 into 1/2

	// fourth, last move
	if( ts->last_move==0 || ts->last_move>=64 )
		b->last_move=INVALID_LAST_MOVE;
	else
		b->last_move=(ts->last_move&56)+(b->player==2?2:5);// new format


	// FYI - set hmc to 0
	b->halfmoveclock=0;
}
#else
void convert_TS_to_board(board*,ts_entry*);
#endif

// these vars are being trained
#if ENGINE==0
alignas(64) static float coeffs_1_nna[NnWeights::numInputs][NnWeights::numN1];	// coeffs for first layer
alignas(64) static float coeffs_2_nna[NnWeights::numN1];				// coeffs for second layer

static double l_rate;						// learning rate
static ts_entry2 *ts_all;					// training set data
static double RR0,RR1,RR0a,RR1a;
static unsigned int cc,cca;
static unsigned int pos_count0,ii,batch_size,iter;

typedef struct {
	float out_1_nna[NnWeights::numN1];					// outputs for 1st layer
	float out_lasta_nna;						// outputs for last layer
	unsigned int inp_nn2[64];					// index of up to 64 pieces on the board. First one is bias - always 1. Terminated by 1000. Max length=32+bias+terminator=34.
} data_nn;


inline float RLU(float s){return(std::max<float>(0,s));} // rectified linear unit

void pass_forward2plus(data_nn *d){// compute output of network: populate second and later layers
	float sa;
	unsigned int i;

	for(i=0;i<NnWeights::numN1;++i)
		d->out_1_nna[i]=RLU(d->out_1_nna[i]);

	// process 2nd layer
	sa=0;
	for(i=0;i<NnWeights::numN1;++i){
		sa+=coeffs_2_nna[i]*d->out_1_nna[i];
	}
	d->out_lasta_nna=sa;
}
#endif


#if ENGINE==0
inline unsigned int piece_index(unsigned char p0,unsigned int sq,unsigned int symm){// symm=flips: 0=as is, 1=w-b, 2=l-r, 3=w-b and l-r
	static const unsigned char r[]=  {0, 0, 1, 2, 3, 4, 5,0,0, 6, 7, 8, 9, 10, 11}; // piece decode (0-11)
	//								  e  P  N  B  R  Q  K e e  p  n  b  r   q   k	
	
	// exchange b/w if symm is odd
	unsigned char p=r[p0^((symm&1)<<3)];

	unsigned int k=1+p+flips[sq][symm]*12;
	assert(k<num_inputs);
	return(k);
}

static Spinlock l2;		// spinlock

void normalize_coeffs(void){
	//return;//
	float cc[num_inputs][num_n1];
	int i,j,j1;

	// black/white: make last 32 b/w flip of first 32. That is, make both average of them.
	for(i=0;i<num_n1/2;++i){
		cc[0][i+num_n1/2]=cc[0][i]=(coeffs_1_nna[0][i]+coeffs_1_nna[0][i+num_n1/2])/2;
		for(j=1;j<num_inputs;j++){// skip constant
			int c=int( ( (j-1)%12 )/6 );// color
			int p=(j-1)%6; // piece
			int sq=(j-1)/12; // square
			//j=1+sq*12+p+c*6;
			j1=1+flips[sq][1]*12+p+(1-c)*6;//1=w/b
			cc[j][i]=(coeffs_1_nna[j][i]+coeffs_1_nna[j1][i+num_n1/2])/2;// temp=(c+color(c))/2
			cc[j1][i+num_n1/2]=cc[j][i];
		}
		coeffs_2_nna[i]=(coeffs_2_nna[i]-coeffs_2_nna[i+num_n1/2])/2;
		coeffs_2_nna[i+num_n1/2]=-coeffs_2_nna[i];
	}
	memcpy(coeffs_1_nna,cc,sizeof(coeffs_1_nna));

	// left/right: make last 16-31 l/r flip of first 16. That is, make both average of them. Then translate these to last 32
	for(i=0;i<num_n1/4;++i){
		cc[0][i+num_n1/4]=cc[0][i]=(coeffs_1_nna[0][i]+coeffs_1_nna[0][i+num_n1/4])/2;
		for(j=1;j<num_inputs;j++){// skip constant
			int c=int( ( (j-1)%12 )/6 );// color
			int p=(j-1)%6; // piece
			int sq=(j-1)/12; // square
			//j=1+sq*12+p+c*6;
			j1=1+flips[sq][2]*12+p+c*6;//2=l-r
			cc[j][i]=(coeffs_1_nna[j][i]+coeffs_1_nna[j1][i+num_n1/4])/2;// temp=(c+l/r(c))/2
			cc[j1][i+num_n1/4]=cc[j][i];
		}
		coeffs_2_nna[i]=(coeffs_2_nna[i]+coeffs_2_nna[i+num_n1/4])/2;
		coeffs_2_nna[i+num_n1/4]=coeffs_2_nna[i];
	}
	// copy first 32 into second 32
	for(i=0;i<num_n1/2;++i){
		cc[0][i+num_n1/2]=cc[0][i];
		for(j=1;j<num_inputs;j++){// skip constant
			int c=int( ( (j-1)%12 )/6 );// color
			int p=(j-1)%6; // piece
			int sq=(j-1)/12; // square
			//j=1+sq*12+p+c*6;
			j1=1+flips[sq][1]*12+p+(1-c)*6;//1=w/b
			cc[j1][i+num_n1/2]=cc[j][i];
		}
		coeffs_2_nna[i+num_n1/2]=-coeffs_2_nna[i];
	}
	memcpy(coeffs_1_nna,cc,sizeof(coeffs_1_nna));
}

void update_coeffs(float *coeffs_1_nn_la,float *coeffs_2_nn_la,int type){
	// include L2 regularization: reduce all weights by % per position.
	float X=float(1.-l_rate*double(batch_size)*.5); // was 0.1
	unsigned int j;

	// momentum
	const float mm=0.9f; // coeff of momentum
	static float coeffs_1_nn_lma[num_n1][num_inputs];
	static float coeffs_2_nn_lma[num_n1];
	

	if( type>10 ){
		memset(coeffs_1_nn_lma,0,num_n1*num_inputs*sizeof(float));
		memset(coeffs_2_nn_lma,0,num_n1*sizeof(float));
		return;
	}

	l2.acquire();
	for(j=0;j<num_inputs*num_n1;++j)
		coeffs_1_nna[0][j]=X*coeffs_1_nna[0][j]+mm*coeffs_1_nn_lma[0][j]+coeffs_1_nn_la[j];
	for(j=0;j<num_n1;++j){
		float t=X*coeffs_2_nna[j]+mm*coeffs_2_nn_lma[j]+coeffs_2_nn_la[j];
		coeffs_2_nna[j]=t;
	}
	normalize_coeffs();
	l2.release();


	memcpy(coeffs_1_nn_lma,coeffs_1_nn_la,num_n1*num_inputs*sizeof(float));
	memcpy(coeffs_2_nn_lma,coeffs_2_nn_la,num_n1*sizeof(float));
	

	memset(coeffs_1_nn_la,0,num_n1*num_inputs*sizeof(float));
	memset(coeffs_2_nn_la,0,num_n1*sizeof(float));
}

void obj(double,double,double*);
#define excl_res 3
static DWORD WINAPI eval_RR(PVOID p){// 1 thread loops over positions to get RR only
	data_nn d_nn;
	double RR0_l=0.,RR1_l=0.,RR0a_l=0.,RR1a_l=0.;
	unsigned int iil,j,l,n,cc_l=0,cca_l=0;

	unsigned int bb=100000; // 100K - large chunks
	while(1){// infinite loop
		iil=InterlockedExchangeAdd((LONG*)&ii,bb); // this is equivalent to locked "i=ii; ii+=X;", but is a lot faster with many threads.
		if( iil>=pos_count0 ) break;	// exit
		for(unsigned int tt=0;tt<bb;++tt){
		if( iil+tt>=pos_count0 ) break;	// exit
		ts_entry2 *tsep=&ts_all[iil+tt];

		// create inputs.
		unsigned int symm=0; // 0-3, randomly selected.*******************************************************************
		unsigned int player=tsep->player;
		if( symm&1 ) player^=1; // 0<->1 if odd symm
		int score_shallow,score_deep;
		if( player==1 ){//change sign of scores if black is to move. Now score is for white
			score_shallow=-tsep->score_shallow;
			score_deep=-tsep->score_deep;
		}else{
			score_shallow=tsep->score_shallow;
			score_deep=tsep->score_deep;
		}
		for(n=0;n<num_n1;++n)
			d_nn.out_1_nna[n]=coeffs_1_nna[0][n]; // bias in
		for(j=0;j<64;j+=2){// loop over all squares. 2 sqs are processed in 1 iteration.
			// cell values are: 0=empty, 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king. Plus 8 if black(top bit). So, 0-14. Use 1/2 byte.
			unsigned char v=tsep->piece[j/2],p1;
			p1=v&15;
			if( p1 ){
				l=piece_index(p1,j,symm);
				__m256 *p1=(__m256*)&d_nn.out_1_nna[0],*p2=(__m256*)&coeffs_1_nna[l][0];
				for(n=0;n<num_n1/8;++n) p1[n]=_mm256_add_ps(p1[n],p2[n]);
			}
			p1=v>>4;
			if( p1 ){
				l=piece_index(p1,j+1,symm);
				__m256 *p1=(__m256*)&d_nn.out_1_nna[0],*p2=(__m256*)&coeffs_1_nna[l][0];
				for(n=0;n<num_n1/8;++n) p1[n]=_mm256_add_ps(p1[n],p2[n]);
			}
		}
		pass_forward2plus(&d_nn);
		float s=d_nn.out_lasta_nna;

		// alt version of creating inputs - for testing
		/*board_plus ts_b;
		convert_TS_to_board(&ts_b.b,(ts_entry*)tsep);// set board
		set_bitboards(&ts_b.b);// set bitboards
		int pv=get_piece_value(&ts_b.b);
		ts_b.b.piece_value=pv;
		s=pass_forward_b(&ts_b.b); // already blended *******************************************
		*/

		double res0[3];obj(score_shallow,score_deep,res0);// base R2
		double res[3];obj(score_shallow+s,score_deep,res);// with adjusted score
	
		// second set
		if( ((iil+tt)%5)==excl_res ){
			RR0a_l+=res0[0];
			RR1a_l+=res[0];
			cca_l++;
		}else{// first set
			RR0_l+=res0[0];
			RR1_l+=res[0];
			cc_l++;
		}
		} // close loop over tt
	}// end of the loop over TS
	l2.acquire();
	RR0+=RR0_l;RR1+=RR1_l;
	RR0a+=RR0a_l;RR1a+=RR1a_l;
	cc+=cc_l;cca+=cca_l;
	l2.release();
	return(0);
}

static DWORD WINAPI train_nn(PVOID p){// 1 thread loops over positions.
	data_nn d_nn;
	float coeffs_1_nn_la[num_inputs][num_n1],coeffs_2_nn_la[num_n1];
	unsigned int iil,j,k,l,counter=0,n;
	double RR1_l=0.;
	unsigned int cc_l=0;

	memset(coeffs_1_nn_la,0,num_n1*num_inputs*sizeof(float));memset(coeffs_2_nn_la,0,num_n1*sizeof(float));
	unsigned int bb=(batch_size*5)/4; // incr to account for dropped validation cases
	while(1){// infinite loop
		iil=InterlockedExchangeAdd((LONG*)&ii,bb); // this is equivalent to locked "i=ii; ii+=X;", but is a lot faster with many threads.
		if( iil>=pos_count0 ) break;	// exit
		counter=0;
		for(unsigned int tt=0;tt<bb;++tt){
		unsigned int pi=iil+tt;
		if( pi>=pos_count0 ) break;	// exit
		// second set - drop it
		if( (pi%5)==excl_res )continue;
		ts_entry2 *tsep=&ts_all[pi];

		// create inputs.
		unsigned int symm=rand()&3; // 0-3, randomly selected.*******************************************************************
		unsigned int player=tsep->player;
		if( symm&1 ) player^=1; // 0<->1 if odd symm
		int score_shallow,score_deep;
		if( player==1 ){//change sign of scores if black is to move. Now score is for white
			score_shallow=-tsep->score_shallow;
			score_deep=-tsep->score_deep;
		}else{
			score_shallow=tsep->score_shallow;
			score_deep=tsep->score_deep;
		}

		k=0;d_nn.inp_nn2[k++]=0; // bias1. Only used for training.
		for(n=0;n<num_n1;++n)
			d_nn.out_1_nna[n]=coeffs_1_nna[0][n]; // bias in
		for(j=0;j<64;j+=2){// loop over all squares. 2 sqs are processed in 1 iteration.
			// cell values are: 0=empty, 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king. Plus 8 if black(top bit). So, 0-14. Use 1/2 byte.
			unsigned char v=tsep->piece[j/2],p1;
			p1=v&15;
			if( p1 ){
				l=piece_index(p1,j,symm);
				d_nn.inp_nn2[k++]=l;
				__m256 *p1=(__m256*)&d_nn.out_1_nna[0],*p2=(__m256*)&coeffs_1_nna[l][0];
				for(n=0;n<num_n1/8;++n) p1[n]=_mm256_add_ps(p1[n],p2[n]);

			}
			p1=v>>4;
			if( p1 ){
				l=piece_index(p1,j+1,symm);
				d_nn.inp_nn2[k++]=l;
				__m256 *p1=(__m256*)&d_nn.out_1_nna[0],*p2=(__m256*)&coeffs_1_nna[l][0];
				for(n=0;n<num_n1/8;++n) p1[n]=_mm256_add_ps(p1[n],p2[n]);
			}
		}
		d_nn.inp_nn2[k]=1000;  // terminator
		assert(k<34+2); // this should never be more then 32 pieces+bias+terminator=34.
		pass_forward2plus(&d_nn);
		float s=d_nn.out_lasta_nna;


		// alt version of creating inputs - for testing
		/*board_plus ts_b;
		convert_TS_to_board(&ts_b.b,(ts_entry*)tsep);// set board
		set_bitboards(&ts_b.b);// set bitboards
		int pv=get_piece_value(&ts_b.b);
		ts_b.b.piece_value=pv;
		s=pass_forward_b(&ts_b.b); //  *******************************************
		if( symm&1 ) s=-s; // need this!*/


		// with adjusted score
		double res[3];obj(score_shallow+s,score_deep,res);
		RR1_l+=res[0];
		cc_l++;
		counter++;
		double aa,ba=-l_rate*res[1]; // o' times learning rate.

		// adjust last layer coeffs - deriv is always 1.
		for(j=0;j<num_n1;++j){
			coeffs_2_nn_la[j]+=float(ba*d_nn.out_1_nna[j]);// no activation function here, deriv=1.
			aa=coeffs_2_nna[j];							// deriv of last layer (no activation function), so coeff only
			if( fabs(d_nn.out_1_nna[j])>1e-10 ){// deriv of second layer: 0/1 for RLU
				aa*=ba;									// put error/learning rate here

				// adjust first layer coeffs
				k=0;
				while( (l=d_nn.inp_nn2[k++])<1000 )
					coeffs_1_nn_la[l][j]+=float(aa); // here weight is always 1. This is the slowest part of training.
			}
		}
		} // close loop over tt

		// update global coeffs?
		if( counter){
			update_coeffs((float*)coeffs_1_nn_la,(float*)coeffs_2_nn_la,0);
			counter=0; // reset
		}
	}// end of the loop over TS
	if( counter ) update_coeffs((float*)coeffs_1_nn_la,(float*)coeffs_2_nn_la,0); // update coeffs on exit
	l2.acquire();
	RR1+=RR1_l;
	cc+=cc_l;
	l2.release();
	return(0);
}

int fmc_p_sort_function_hash(const void*, const void*);
void get_shallow_score(void){// run over TS, get shallow score, save as TS2. In eval, exclude nn call. And set "train" to 1 (because otherwise eval hash returns wrong values, since i don't define hash key).
	unsigned int i,j;
	assert(TRAIN);

	// load old TS
	FILE *f=fopen("c://xde//chess//data//TSn2.bin","rb");// main TS file, after Qs. Now just call eval on it.
	fread(&pos_count0,sizeof(unsigned int),1,f);
	ts_all=(ts_entry2*)malloc(sizeof(ts_entry2)*pos_count0); // storage of ts entries - 40 bytes each.
	pos_count0=(unsigned int)fread(ts_all,sizeof(ts_entry2),pos_count0,f);
	fclose(f);

	// get scores
	for(j=i=0;i<pos_count0;++i){
		board_plus ts_b;
		DWORD bit;

		// skip unsolved. This comes into effect.
		if( ts_all[i].remarks<2 ) continue;

		// set board
		convert_TS_to_board(&ts_b.b,(ts_entry*)(ts_all+i));

		// set bitboards
		set_bitboards(&ts_b.b);

		// set king positions
		_BitScanForward64(&bit,ts_b.b.piececolorBB[5][0]);ts_b.b.kp[0]=(unsigned char)bit;// white
		_BitScanForward64(&bit,ts_b.b.piececolorBB[5][1]);ts_b.b.kp[1]=(unsigned char)bit;// black

		// init PST scores
		get_scores(&ts_b.b);

		// get material key
		ts_b.b.mat_key=get_mat_key(&ts_b.b);

		// get hashes
		ts_b.b.pawn_hash_key=get_pawn_hash_key(&ts_b.b); // this is currently needed. But why?

		// init search
		ts_b.b.em_break=0; // reset
		ts_b.b.slave_index=0;
		ts_b.score_shallow=eval(&ts_b.b); // call eval directly. Here call to NN should be commented out! ****************************************
		if( abs(ts_b.score_shallow)>2000 ) continue; // skip if search sees a checkmate. This comes into effect.
		if( popcnt64l(ts_b.b.colorBB[0]|ts_b.b.colorBB[1])<=5 ) continue;// skip if 5 or fewer pieces
		if( ts_all[i].score_deep>0 ) ts_b.score_deep=5000;// win - +5K
		else if( ts_all[i].score_deep<0 ) ts_b.score_deep=-5000;// loss - -5K
		else ts_b.score_deep=0;// draw

		// add results to output
		ts_all[j]=ts_all[i];
		ts_all[j].score_deep=ts_b.score_deep; // 0, +/-5000, transformed
		ts_all[j].score_shallow=ts_b.score_shallow; // not transformed!
		j++;
	}
	pos_count0=j;// save new pos count

	// sort by hash, to simulate random distribution
	qsort(ts_all,pos_count0,sizeof(ts_entry),fmc_p_sort_function_hash);

	// determine R2
	double RR0=0;
	for(i=0;i<pos_count0;++i){
		ts_entry2 tse=ts_all[i];
		double res0[3];obj(tse.score_shallow,tse.score_deep,res0);
		RR0+=res0[0];
	}

	// save new TS
	f=fopen("c://xde//chess//data//TSn2_s.bin","wb");
	fwrite(&pos_count0,sizeof(unsigned int),1,f);
	fwrite(ts_all,sizeof(ts_entry2),pos_count0,f);
	fclose(f);
	exit(0);
}

inline float rr(void){
	static const float v[10]={-1.644853,-1.036432877,-0.674490366,-0.385321073,-0.125661472,0.125661472,0.385321073,0.674490366,1.036432877,1.644853};// normal
	unsigned int index=unsigned int(rand()*10./(1.+RAND_MAX));
	float s=v[index];
	return(s);
}

void write_coeffs(int d){// write coeffs to file
	unsigned int i,j;
	char format[]="%.2f,";
	format[2]=(d%10)+'0'; // number of digits to print

	// scale all L1 coeffs so that L2 is +-1
	for(i=0;i<num_n1;++i){
		float F=float(fabs(coeffs_2_nna[i]));
		for(j=0;j<num_inputs;++j)
			coeffs_1_nna[j][i]=F*coeffs_1_nna[j][i];
		coeffs_2_nna[i]/=F;
	}
	float f1=0;
	for(i=0;i<num_n1;++i)
		f1+=coeffs_2_nna[i];
	
	// add bias to king coeffs: WK for +, BK for-
	for(i=0;i<num_n1;++i){// 769 index is (0-5)+6*black+sq*12+1(bias)
		j=1+5+(i>=32?6:0);
		for(unsigned int k=0;k<64;++k)// k loops over 64 squares on the boeard
			coeffs_1_nna[j+k*12][i]+=coeffs_1_nna[0][i];
		coeffs_1_nna[0][i]=0;
	}

	// save int coeffs
	unsigned int counter=0;
	FILE *f=fopen("c://xde//chess//out//ci.csv","w");
	for(j=2;j<num_inputs;++j)// skip bias and WP sq=0 - they are both empty
		for(i=0;i<num_n1;++i) {
			fprintf(f,"%.0f,",coeffs_1_nna[j][i]);
			if( ++counter>=64 ){
				counter=0;
				fprintf(f,"\n");
			}
		}
	fprintf(f,"\n\n");
	if( fabs(f1)>0.1f ){
		fprintf(f,"\n\nbad L2 coeffs!!!\n");
		for(i=0;i<num_n1;++i) fprintf(f,"%.2f,",coeffs_2_nna[i]);
		fprintf(f,"\n");
	}
	fclose(f);

	// second, binary
	if( d<10) f=fopen("c://xde//chess//out//nn_log1.bin","wb");
	else f=fopen("c://xde//chess//out//nn_log2.bin","wb"); // test - new file
	fwrite(coeffs_1_nna,sizeof(coeffs_1_nna),1,f);
	fwrite(coeffs_2_nna,sizeof(coeffs_2_nna),1,f);
	fclose(f);
}

void read_nn_coeffs(void){// read NN coeffs from file
	unsigned int i,j;
	
	// init NN coeffs to small random values
	for(j=0;j<num_n1;++j) for(i=0;i<num_inputs;++i)
		coeffs_1_nna[i][j]=float(rr()/sqrt(20.));
	for(j=0;j<num_n1;++j)
		coeffs_1_nna[0][j]=0.1f; // L1 bias
	for(i=0;i<num_n1/2;++i)
		coeffs_2_nna[i]=1.0f;// split sign: produces higher and more compact R2 distribution
	for(;i<num_n1;++i)
		coeffs_2_nna[i]=-1.0f;
	normalize_coeffs();

	/*FILE *f=fopen("c://xde//chess//out//nn_log1.bin","rb");
	if( f!=NULL ){
		//fread(coeffs_1_nna,sizeof(coeffs_1_nna),1,f);
		fread(coeffs_1_nna,769*64*4,1,f);
		fread(coeffs_2_nna,sizeof(coeffs_2_nna),1,f);
		fclose(f);
	}*/

	// Empty cells are 1,7,85,91,97,103, etc
	for(j=0;j<8;j++) // loop over all files
		for(i=0;i<num_n1;++i){// loop over all N1
			coeffs_1_nna[1+12*(j*8+0)+0][i]=0;// WP, rank 1
			coeffs_1_nna[1+12*(j*8+7)+0][i]=0;// WP, rank 8
			coeffs_1_nna[1+12*(j*8+0)+6][i]=0;// BP, rank 1
			coeffs_1_nna[1+12*(j*8+7)+6][i]=0;// BP, rank 8
		}
}

inline unsigned char pp(ts_entry2* tse,unsigned int sq){
	if( (sq&1) ) return(tse->piece[sq/2]>>4); // odd, top
	else return(tse->piece[sq/2]&15); // even, bottom
}

//double run_nn1(void){// main function
void run_nn(void){// main function
	unsigned int j;
	DWORD t1=get_time();
	srand(t1);
	update_coeffs((float*)NULL,(float*)NULL,100);// reset	
	read_nn_coeffs();// read coeffs from file (or init to random if file does not exist)
	
	// try load updated file
	FILE *f=fopen("c://xde//chess//data//TSn2_s.bin","rb");
	//FILE *f=fopen("c://xde//chess//data//TSn2_s4.bin","rb");
	if( f!=NULL ){
		// load new TS
		fread(&pos_count0,sizeof(unsigned int),1,f);
		ts_all=(ts_entry2*)malloc(sizeof(ts_entry2)*pos_count0); // storage of ts entries - 40 bytes each.
		pos_count0=(unsigned int)fread(ts_all,sizeof(ts_entry2),pos_count0,f);
		fclose(f);
	}else
		get_shallow_score(); // create new TS and exit


	// train the NN
	#define nn_threads 12					// total number of calc threads: 12
	l_rate=2e-9;							// learning rate: 5e-12,3e-10,2e-9
	batch_size=10000;						// batch size: 10K
	f=fopen("c://xde//chess//out//nn_log.csv","w");
	fprintf(f,"i,RR0,RR1,RR0a,RR1a,RR1t,dRR,dRRa,l_rate,batch_size,time,k\n");
	fclose(f);
	HANDLE h[nn_threads+1];	// calculation thread handle
	double RR1t=0;
	for(iter=0;iter<200;iter++){
	
		// run training
		l2.release();
		if( fabs(l_rate)>1e-20 ){// skip training if no learning
			cc=cca=0;RR0=RR1=RR0a=RR1a=0.; // reset
			ii=0;
			for(j=0;j<nn_threads-1;++j) h[j]=CreateThread(NULL,0,train_nn,(PVOID)j,0,NULL);//sec.attr., stack, function, param,flags, thread id
			train_nn((PVOID)j);
			WaitForMultipleObjects(nn_threads-1,h,TRUE,INFINITE);// wait for threads to terminate.
			for(j=0;j<nn_threads-1;++j)  if( h[j] ) CloseHandle(h[j]);
			RR1t=sqrt(RR1/cc);

			// scale all L1 coeffs so that L2 is +-1
			/*for(unsigned int i=0;i<num_n1;++i){
				float F=float(fabs(coeffs_2_nna[i]));
				for(j=0;j<num_inputs;++j)
					coeffs_1_nna[j][i]=F*coeffs_1_nna[j][i];
				coeffs_2_nna[i]/=F;
			}*/
		}

		// evaluate R2
		cc=cca=0;RR0=RR1=RR0a=RR1a=0.; // reset
		ii=0;
		for(j=0;j<nn_threads-1;++j) h[j]=CreateThread(NULL,0,eval_RR,(PVOID)j,0,NULL);//sec.attr., stack, function, param,flags, thread id
		eval_RR((PVOID)j);
		WaitForMultipleObjects(nn_threads-1,h,TRUE,INFINITE);// wait for threads to terminate.
		for(j=0;j<nn_threads-1;++j)  if( h[j]!=NULL ) CloseHandle(h[j]);
		RR0=sqrt(RR0/cc);RR1=sqrt(RR1/cc);RR0a=sqrt(RR0a/cca);RR1a=sqrt(RR1a/cca);

		int k=-12;
		for(j=0;j<num_n1/2;++j){
			if( coeffs_2_nna[j]>0 ) k++;
		}
		f=fopen("c://xde//chess//out//nn_log.csv","a");
		fprintf(f,"%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2g,%d,%d,%d\n",iter,RR0,RR1,RR0a,RR1a,RR1t,RR1-RR0,RR1a-RR0a,l_rate,batch_size,(get_time()-t1)/1000,k);
		fclose(f);

		// adjust parameters
		if( iter>30 ) l_rate*=.98;
	}
	if( iter>2 ) write_coeffs(12); // write coeffs to file
	free(ts_all);
	exit(0);
	//return(-RR1a+RR0a);
}

/*void run_nn(void){
	FILE *f;
	double r,r0=0;;
	unsigned int i;

	f=fopen("c://xde//chess//out//ll.csv","w");
	fprintf(f,"starting...\n");
	fclose(f);
	for(i=0;i<200;++i){
		r=run_nn1();
		f=fopen("c://xde//chess//out//ll.csv","a");
		fprintf(f,"r=%.3f\n",r);
		if( r>r0 ){
			r0=r;
			fprintf(f,",saving new best result\n");
			write_coeffs(12);
		}	
		fclose(f);
	}
	exit(5);
}*/
#endif
