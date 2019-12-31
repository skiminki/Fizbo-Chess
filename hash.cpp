// transposition table implementation

#include "hash.h"
#include <cstdlib>
#include <cstring>
#include <random>

#include <atomic>

#include "chess.h"

std::atomic<HashSlotStorageType> *h; // hash table

namespace {

	struct hash {
		uint32_t ttHashKey : ttHashKeyBits;     // 26 bits of hash key
		uint8_t depth:6;			// 6 bits of depth: 0 to 63 = 32
		int16_t score;				// 16 bits, score, +-10K. Need 15 bits to cover +-16K. = 48
		uint8_t from:6;			// "from". Need 6 bits = 54
		uint8_t type:2;			// score type: 0/1/2=exact,lower,upper. Need 2 bits. = 56
		uint8_t to:6;				// "to". Need 6 bits. = 62
		uint8_t age:2;			// age: 0 to 3. Need 2 bits. = 64
	}; // 8 bytes.

	static_assert(sizeof(hash) == sizeof(HashSlotStorageType), "Cooked and storage types must match");

	union HashSlotCookedAndRawUnion {
		hash cooked;
		uint64_t raw;
	};

	inline HashSlotStorageType ttSlotToU64(hash h)
	{
		HashSlotCookedAndRawUnion tt;
		tt.cooked = h;
		return tt.raw;
	}

	inline hash ttSlotFromU64(HashSlotStorageType u)
	{
		HashSlotCookedAndRawUnion tt;
		tt.raw = u;
		return tt.cooked;
	}
}

UINT64 *mem;
UINT64 hash_index_mask;
unsigned int TTage;// TT aging counter, 0 to 3.
unsigned int HBITS=24; // option**** 16 Mil entries * 8 bytes= 128 Mb. 24 bits. Main hash is 22 bits of 4-way blocks.
//unsigned int HBITS=27; // option**** 128 Mil entries * 8 bytes= 1 Gb. 27 bits. Main hash is 25 bits of 4-way blocks.
static unsigned int HSIZE;

namespace {

	// We'll use the Visual C++ PRNG to get the same Zobrist keys for Fizbo-Linux
	// and Fizbo on Windows. Here it is:
	ZobristArrayType initZobristKeys()
	{
		using VisualCppPRNG = std::linear_congruential_engine<uint_fast64_t, 214013, 2531011, uint64_t { 1 } << 32>;
		VisualCppPRNG msRandom;

		msRandom.seed(1678579445);

		ZobristArrayType ret { };

		for (size_t j = 0; j < 2; ++j) // player
			for (size_t k = 0; k < 6; ++k) // piece
				for (size_t i = 0; i < 64; ++i){ // square
					uint64_t z = (msRandom() >> 16) & 0x7FFFU; // 1
					z = (z<<15) ^ ((msRandom() >> 16) & 0x7FFFU); // 2
					z = (z<<15) ^ ((msRandom() >> 16) & 0x7FFFU); // 3
					z = (z<<15) ^ ((msRandom() >> 16) & 0x7FFFU); // 4
					z = (z<<15) ^ ((msRandom() >> 16) & 0x7FFFU); // 5
					ret[k][j][i]=z; // 15 bits each,x5= 75 bit total
				}

		// set pawn zorb to queen for promotion, so that i don't have to change it. But wil have to change pawn hash!
		for (size_t i = 0; i < 64; i += 8 ){// square (in steps of 8)
			ret[0][0][i+7]=ret[4][0][i+7];
			ret[0][1][i]=ret[4][1][i];
		}

		return ret;
	}
}

// zobrist keys: 12 pieces by 64 cells
const ZobristArrayType zorb { initZobristKeys() };

void int_m2(void);
extern UINT64 *eh; // eval hash pointer
extern short int *mh; // material table pointer
void clear_hash(unsigned int i){//0: TT only. >0: Pawn hash also.
	memset(static_cast<void *>(h), 0, sizeof(hash)*HSIZE);// TT hash
	// Eval hash is always reset in solve_prep function.
	if(i) memset(ph,0,8*PHSIZE);// pawn hash
}

void init_hash(void){
	static UINT64 *meml=NULL;
	static unsigned int HBL=0,virt=0;

	// alloc memory
	if( HBITS!=HBL ){
		if( h ){// release already allocated memory
			if( virt==0 ){
				free(meml);free(h);free(ph);
				free(eh);free(mh);
			}
			#if USE_VIRT_MEM
			else
				VirtualFree(h,0,MEM_RELEASE);
			#endif
			ph=NULL;h=NULL;
		}
		HSIZE=UINT64(1)<<HBITS;
		hash_index_mask=UINT64(0x0fffffffffffffff)>>(64-4+2-HBITS);
		#if USE_VIRT_MEM
		HANDLE token_handle;
		TOKEN_PRIVILEGES tp;
		SIZE_T MinimumPSize = GetLargePageMinimum();
		if( MinimumPSize ){
			#if ALLOW_LOG
			fprintf(f_log,"Large Pages supported\n");
			fprintf(f_log,"Minimum Large Pages size = %i\n", (unsigned int)MinimumPSize);
			#endif

			OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token_handle);
			LookupPrivilegeValue(NULL, TEXT("SeLockMemoryPrivilege"), &tp.Privileges[0].Luid);
			tp.PrivilegeCount = 1;
			tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
			AdjustTokenPrivileges(token_handle, false, &tp, 0, NULL, 0);
			CloseHandle(token_handle);

			SIZE_T size=sizeof(hash)*HSIZE+sizeof(UINT64)*PHSIZE+sizeof(UINT64)*EHSIZE+sizeof(short)*256*1024+2*1024*1024;// raw size(TT, pawns, eval, material)+2Mb
			size=(((size-1)/MinimumPSize)+1)*MinimumPSize; // round up to next large page
			#if ALLOW_LOG
			fprintf(f_log,"Allocating %u Mb\n",(unsigned int)(size/1024/1024));
			#endif
			virt=1; // mark it as virtually allocated
			h=(hash*)VirtualAlloc(NULL, size, MEM_LARGE_PAGES | MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);// use virtual alloc;
			ph=(UINT64*)(h+HSIZE);
			eh=(UINT64*)(ph+PHSIZE);
			mh=(short*)(eh+EHSIZE);
			meml=mem=(UINT64*)(mh+256*1024);// free 2 Mb of memory. Put large arrays here.
			if( h==NULL ){// cannot allocate. Why?
				LPVOID lpMsgBuf;
				DWORD dw = GetLastError(); 
				FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,NULL,dw,MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),(LPTSTR) &lpMsgBuf,0, NULL );
				#if ALLOW_LOG
				fprintf(f_log,"Memory alloc failed with error %d: %s\n",dw,(char*)lpMsgBuf);
				#endif
				LocalFree(lpMsgBuf);
			}else{
				#if ALLOW_LOG
				fprintf(f_log,"Large pages were set successfully\n");
				#endif
			}
		}
		#endif
	
		if( h==NULL ){// cannot allocate. Use regular malloc
			h=(std::atomic<HashSlotStorageType> *)malloc(sizeof(hash)*HSIZE);// raw size
			ph=(UINT64*)malloc(8*PHSIZE);// raw size
			eh=(UINT64*)malloc(8*EHSIZE);// raw size
			mh=(short int*)malloc(512*1024);// raw size
			meml=mem=(UINT64*)malloc(2*1024*1024);// raw size
			virt=0;
			#if ENGINE
			if( h==NULL || mem==NULL ) pass_message_to_GUI("info string Memory alloc failed\n");
			#endif
			#if ALLOW_LOG
			if( h!=NULL && mem!=NULL ) fprintf(f_log,"Small pages were set successfully\n");
			#endif
		}
		clear_hash(1);
		if( HBL){
			int_m2();		// init magic bitboards, on second call only
			init_material();// init material
		}
		HBL=HBITS; //save
	}
}

unsigned int hashfull(void){// cound hash entries in the first 1000 spots, for current age only
	unsigned int i,s;
	for(i=s=0;i<1000;++i) {
		hash entry { ttSlotFromU64(h[i].load(std::memory_order_relaxed)) };

		if( entry.age == TTage )
			s++;
	}
	return(s);
}

NOINLINE unsigned int lookup_hash(unsigned int depth, const board *b, hash_data *hd, unsigned int ply){// returns indicator only.
	std::atomic<HashSlotStorageType> *h1 = &h[getHashIndex(b->hash_key)];// always start at the beginiing of block of 4 - 4-way set-associative structure.
	hash h_read;
	unsigned int i; //,lock4a=(((unsigned int *)&b->hash_key)[1])<<6;// shift out top 6 bits - they are depth.

	const uint32_t ttHashKey = ((b->hash_key) >> 32) & 0x03FFFFFFU; // 26 bits of hash key

	// check if it matches all pieces and player
	for(i=0;i<4;++i,++h1){
		h_read = ttSlotFromU64(h1->load(std::memory_order_relaxed));

		if (h_read.ttHashKey == ttHashKey) { // match
			hd->depth=h_read.depth;								// return depth
			hd->bound_type=h_read.type;
			hd->tt_score=h_read.score;
			hd->move[0] = h_read.from;
			hd->move[1] = h_read.to;
			hd->alp=MIN_SCORE;
			hd->be=MAX_SCORE;					// init score
			h1->store(ttSlotToU64(h_read), std::memory_order_relaxed); // not stale anymore

			if( depth<=h_read.depth ){			// only use bounds for search of the same OR GREATER depth.
				int s1=h_read.score;			// score to be returned
				if( abs(s1)>=5000 ){	    	// mate - adjust by ply
					if( s1<0 )// make score smaller
						s1+=ply;
					else
						s1-=ply;
				}
				if( h_read.type==0 )			//exact score
					hd->be=hd->alp=s1;			// assign both bounds
				else if( h_read.type==1 )		//lower bound
					hd->alp=s1;					// assign lower bound
				else							// upper bound
					hd->be=s1;					// assign upper bound
			}
			return(1);// return match
		}
	}
	return(0); // not a match
}

NOINLINE void add_hash(int alp,int be,int score,unsigned char *move,unsigned int depth,const board *b,unsigned int ply){
	static const unsigned int TTage_convert[4][4]={{0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0}};// converts current (first arg) and old (last arg) TT age into age difference
	std::atomic<HashSlotStorageType> *h2;
	std::atomic<HashSlotStorageType> *h1 = &h[getHashIndex(b->hash_key)];// always start at the beginiing of block of 4 - 4-way set-associative structure.
	hash h_read;
	unsigned int i;

	const uint32_t ttHashKey = ((b->hash_key) >> 32) & 0x03FFFFFFU; // 26 bits of hash key

	int s_replace=1000,s1=score; // this is score that i store in TT

	if( abs(s1)>=5000 ){// mate - adjust by ply
		if( s1>0 )// make score larger - undo the effect of reducing loss by ply.
			s1+=ply;
		else
			s1-=ply;
	}

	// check if it matches all pieces and player
	for(i=0;i<4;++i,++h1){
		h_read = ttSlotFromU64(h1->load(std::memory_order_relaxed)); // atomic read
		if (h_read.ttHashKey == ttHashKey) { // match
 			// always replacing is +5 ELO. 6/2017.
			h_read.score=s1;		// score

			if (move[0] || move[1]) {
				// only replace the move if it is valid
				h_read.from = move[0];
				h_read.to = move[1];
			}

			h_read.age=TTage;		// not stale anymore

			if(score<be){
				if(score>alp)
					h_read.type=0;
				else
					h_read.type=2;	// score<=alp - upper bound. Score is in the range -inf, score.
			} else
				h_read.type=1;		// score>=be - lower bound. Score is in the range score, +inf.
			h_read.depth=depth;
			h1->store(ttSlotToU64(h_read), std::memory_order_relaxed); // atomic write

			return; // match found and updated - return
		}else{// no match. Consider it for replacement
			int s=int(h_read.depth)-(int(TTage_convert[TTage][h_read.age])<<8)+(h_read.type==0?2:0);  // order: age(decr), then depth(incr), then node type. Here 2 for PV seems best
			//int s=int(h_read.depth)-(int(TTage_convert[TTage][h_read.age])<<3)+(h_read.type==0?1:0);  // order: age(decr), then depth(incr), then node type: a wash, do not use. 6/2017.
			if( s<s_replace ){// lower depth (or stale, or both) found, record it as best. Here i is already a tie-breaker.
				s_replace=s;
				h2=h1;
			}
		}
	}

	// Always replace - should be better in long games.
	h_read.ttHashKey = ttHashKey;
	h_read.score=s1;
	h_read.from = move[0];
	h_read.to = move[1];
	h_read.age=TTage;	// not stale anymore
	if(score<be){
		if(score>alp)
			h_read.type=0;
		else
			h_read.type=2;
	}else
		h_read.type=1;
	h_read.depth=depth;
	h2->store(ttSlotToU64(h_read), std::memory_order_relaxed); // atomic write
 }

NOINLINE void delete_hash_entry(board *b){// find this position and delete it from TT
	std::atomic<HashSlotStorageType> *h1=&h[getHashIndex(b->hash_key)];// always start at the beginiing of block of 4 - 4-way set-associative structure.

	for (unsigned int i=0; i<4; ++i)
		h1[i].store(0, std::memory_order_relaxed);
 }
