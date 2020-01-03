#ifndef FIZBO_HASH_H_INCLUDED
#define FIZBO_HASH_H_INCLUDED

#include <atomic>
#include <cstddef>
#include <cstdint>

// forward declarations
struct board;

// hash table
using HashSlotStorageType = uint64_t;       // we use this as the storage format for an hash slot to guarantee atomic load/store
extern std::atomic<HashSlotStorageType> *h;

extern uint64_t hash_index_mask;

// main transposition table data structure
constexpr size_t ttHashKeyBits { 26 };
constexpr uint32_t ttHashKeyMask { (uint32_t { 1 } << ttHashKeyBits) - 1 };


// index into the main hash table
// #define get_hash_index (getHashIndex(b->hash_key)) // always start at the beginning of block of 4 - 4-way set-associative structure - corrected

// index into the eval hash table
#define get_eval_hash_index (b->hash_key%EHSIZE)

// size of pawn hash table
#define PHSIZE (1024*512) // 1/2 Mill entries * 8 bytes= 4 Mb. Fits in L3 cache.
//#define PHSIZE (1024*1024*4) // 4 Mill entries * 8 bytes= 32 Mb. For internet games and TCEC.

// size of eval hash table
#define EHSIZE (1024*512) // 1/2 Mill entries * 8 bytes= 4 Mb. Fits in L3 cache. Increasing this significantly improves runtime!
//#define EHSIZE (1024*1024*32) // 32 Mill entries * 8 bytes=256 Mb. For internet games and TCEC.

// transposition table return data structure
typedef struct{
	int alp;
	int be;
	int tt_score;
	unsigned char move[2];		// from, to
	unsigned char depth;
	unsigned char bound_type;	// score type: 0/1/2=exact,lower,upper.
} hash_data; // 16 bytes

void init_hash(uint32_t megabytes);
void clear_hash(unsigned int);
unsigned int lookup_hash(unsigned int depth, const board *b, hash_data *hd, unsigned int ply);
void add_hash(int,int,int,unsigned char*,unsigned int,const board*,unsigned int);
unsigned int hashfull(void);

inline size_t getHashIndex(uint64_t hashKey)
{
	return (hashKey & hash_index_mask) << 2;
}

#endif
