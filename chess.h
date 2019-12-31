
#ifndef FIZBO_CHESS_H_INCLUDED
#define FIZBO_CHESS_H_INCLUDED

#include <array>
#include <cstdint>

#include "bitutils.h"
#include "os-compat.h"

using UINT64 = std::uint64_t;

#include <stdio.h>
#include <assert.h>

#define ENGINE 1		// 1=run as engine, 0=run as windowed interface
#define USE_EGTB 0		// 1
#define ALLOW_LOG 0
#define calc_pst 0		// 1=calculate in eval, 0=update incrementally. Here 0 is slightly faster - use that.

constexpr uint64_t player_zorb { 0xab42094fee35f92eU };

#ifdef __AVX__
#define USE_AVX 1		// this is 5% faster. Also turn off AVX2 compiler switch.
#define USE_PEXT 1
#else
#define USE_AVX 0
#define USE_PEXT 0
#endif

#define SPS_CREATED_NUM_MAX 22 // max SPs created by 1 thread

// portable
#define blsr64l(a) BitUtils::blsr(static_cast<uint64_t>(a))
#define blsr32l(a) BitUtils::blsr(static_cast<uint32_t>(a))

//#define popcnt64l(a) __popcnt64(a)		  // instruction
#define BSF64l(a,b) BitScanForward64(a,b) // instruction
#define BSR64l(a,b) BitScanReverse64(a,b) // instruction

#define popcnt64l(a) BitUtils::popcount(static_cast<uint64_t>(a)) // legacy: +13% run time.

#define TRAIN 0 		// 1=logic for training set - no hashing, record all positions. In training, turn-off EGBB.
#define USE_VIRT_MEM 0	// 1 portable. +3.8% run time.

#define last_move_hash_adj zorb[5][0][(b->last_move&56)+7] // only use file of last move
enum EvalType {Full,NoQueens};// Different eval types, used as template parameter

#define get_time getTimeMs

#if ALLOW_LOG
#if ENGINE
	#define LOG_FILE1 "/tmp/log_e.csv" // main log file for the engine
#else
	#define LOG_FILE1 "c://xde//chess//out//log_i.csv" // main log file for the interface
#endif
#endif

#define EGTBP 6				// 5 or 6 piece endgame tables. Use 6 only for local internet games
#define MAX_SCORE 20000
#define MIN_SCORE -20000
#define GET_BIT(a) BSF64l(&bit,a);a=blsr64l(a);// reset lowest set bit. Works with BitScanForward, not with BitScanReverse!
#define GET_BIT2(a) BSF64l(&bit2,a);a=blsr64l(a);// reset lowest set bit. Works with BitScanForward, not with BitScanReverse!

// play data structure
typedef struct{
	short int stand_pat;
	char ch_ext;		// check extension data
	char cum_cap_val;	// cumulative capture value
	uint8_t to_square;		// "to" square
	#if NDEBUG
	#else
	char cap_val;
	unsigned char from;
	unsigned char to;
	unsigned char move_type;
	#endif
} play; // 3/4 bytes

#define MAX_MOVE_HIST 8			// how many moves to keep in PV history. 8.
#define INVALID_LAST_MOVE 64	// 0 causes false positives with cell==b->last_move logic, so use 64.

/*
// board_light data structure
typedef struct{
	UINT64 hash_key;				// save 1 (8). transposition table key=8 8
	UINT64 pawn_hash_key;			// save 2 (8). pawn transposition table key=8 16
	UINT64 colorBB[2];				// bitboard for all pieces of both players: by player only=16 32
	UINT64 piececolorBB[6][2];		// bitboard for all pieces of both players: by piece and player=96 128
	unsigned char piece[64];		// cell values are: 0=empty, 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king. 3 bits. Plus player as 2 top bits(64 or 128). Store byte=64 192
	unsigned int castle;			// save 3 (8). castling possible: white lower(Q), white upper(K), black lower(q), black upper(k). 1=allowed, 0=not allowed. 196
	short int scorem;
	short int scoree;				// material scores. 200
	unsigned char kp[2];			// save 4 (8). king position for both players. Addressable by player-1. 202
	unsigned char player;			// player. 1/2. 203
	unsigned char last_move;		// last move made to. 204
	unsigned char halfmoveclock;	// for 50 move(100 halfmove) rule 205
	unsigned char nullmove;			// 207
	unsigned char filler[2];		// 208
	unsigned int mat_key;			// save 5 (8). 212
	unsigned int piece_value;		// total piece value, 0 to 64. 216
} board_light; // 216 bytes. Used in training only.
*/


// Square position. a1 = 0, a2 = 1, ..., b1 = 8, b2 = 9,... h8 = 63
using SquareIndex = std::uint8_t;


// board data structure
struct board {
	UINT64 hash_key;				// save 1 (8). transposition table key=8 8
	UINT64 pawn_hash_key;			// save 2 (8). pawn transposition table key=8 16
	UINT64 colorBB[2];				// bitboard for all pieces of both players: by player only=16 32
	UINT64 piececolorBB[6][2];		// bitboard for all pieces of both players: by piece and player=96 128
	unsigned char piece[64];		// cell values are: 0=empty, 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king. 3 bits. Plus player as 2 top bits. Store byte=64 192
	unsigned int castle;			// save 3 (8). castling possible: white lower(Q), white upper(K), black lower(q), black upper(k). 1=allowed, 0=not allowed. 196
	#if calc_pst==0
	union {
		struct {
			int16_t midgame;
			int16_t endgame;				// material scores. 200
		} score;
		uint32_t score_m_and_e;		// accessor for both score fields for optimized +/- when we know that scorem/scoree won't overflow
	};

	#endif
	std::array<SquareIndex, 2> kp;          // save 4 (8). king position for both players. Addressable by player-1. 202
	unsigned char player;			// player. 1/2. 203
	unsigned char last_move;		// last move made to. 204
	unsigned char halfmoveclock;	// for 50 move(100 halfmove) rule 205
	unsigned char nullmove;			// 207
	unsigned char filler[2];		// 208
	unsigned int mat_key;			// save 5 (8). 212
	unsigned int piece_value;		// total piece value, 0 to 64. 216
	unsigned int em_break;			// 220 emergency break for the current thread.
	int history_count[12][64];		// history count table, for move sorting. Player*6+piece, to. 3Kb.
	unsigned short int killer[128][2];// killer move table, for move sorting. Ply, 2 moves, to/from combined into short int. 0.5Kb.
	unsigned short int countermove[12][64];// countermove table, for move sorting. OPP Player*6+piece=12, OPP to. to/from combined into short int. 1.5Kb.
	UINT64 position_hist[100+128];	// 220+1312=1540 Z values for search history. For repetition draws. Indexed by ply. Offset by 100, to capture past positions from the game. 1.3 Kb.
	UINT64 node_count;
	unsigned int max_ply;
	unsigned char move_exclude[2];

	// these should be set by master, and is needed by slaves
	unsigned char sp_level;			// number of split-points above this position
	unsigned char sps_created_num;
	unsigned char sps_created[SPS_CREATED_NUM_MAX];
	void * spp;						// pointer to split-point

	// something that i don't need to copy to slave
	play pl[128];					// offset by 1 just in case. Do not copy last 100 of it to slaves. 128*4=0.5 Kb
	unsigned int slave_index;		// master is 0, slaves are 1+. Do not copy it to slaves.
	unsigned int slave_index0;		// master is 0, slaves are 1+. Do not copy it to slaves.
	unsigned char move_hist[MAX_MOVE_HIST][MAX_MOVE_HIST][2];// move history. Do not copy it to slaves. for 8: 2*8*8=128 bytes
}; // 216 bytes+threading=4,516 if MAX_MOVE_HIST=8.

// data structure for move un-make function
typedef struct{
	UINT64 hash_key;			// 8 transposition table key
	UINT64 pawn_hash_key;			// 16 pawn transposition table key
	unsigned int castle;			// 20 castling possible: white lower(Q), white upper(K), black lower(q), black upper(k). 1=allowed, 0=not allowed.
	uint32_t score_m_and_e;                 // 24
	std::array<SquareIndex, 2> kp;		// 26 king position for both players. Addressable by player-1.
	unsigned char player;			// 27 player. 1/2.
	unsigned char last_move;		// 28 last move made to.
	unsigned char halfmoveclock;	        // 29 for 50 move(100 halfmove) rule
	unsigned char nullmove;                 // 30
	unsigned char padding[2];               // 32
	unsigned int move_type;			// 36 - type of move: 0=quiet; 1=castling; 2=capture, including ep; 4=promotion; 8=ep capture. Combinations are allowed.
	unsigned char w;			// 37 piece captured
	unsigned char cc;			// 38 square where piece was captured (to/to1)
	unsigned char from;			// 39 from
	unsigned char to;			// 40 to
	unsigned int mat_key;			// 44 material key
	unsigned int piece_value;		// 48 total piece value, 0 to 64.
	unsigned char promotion;		// 49 0-Q,1-R,2-B,3-N

	// Note: there will be 7 bytes for alignment on 64-bit architectures

} unmake; // 32+8+4+align=48 bytes

// static_assert(sizeof(unmake) == 56, "the actual size");

// move data structure
typedef struct{
	int score;					// move score. TT move is 1e6. Capture is 200+. Countermove is 195. Killers are 0-100. Quiet moves are negative.
	unsigned char from,to;
} move; // 8 bytes (with alignment)

// best move structure
typedef struct{
	int legal_moves;  // 0/1
	int best_score;
	int alp;
	unsigned char best_move[2];
} best_m; // 16 bytes (with alignment)

// move list data structure
typedef struct{
	UINT64 pinBB;
	UINT64 opp_attacks;			// mask of opp attacks, for selecting losing moves
	unsigned int moves_generated;// 1=captures, 2=non-captures, 32=all
	unsigned int status;		// 0-9=uninitialized; 10-14=initialized(move type identified); 15=TT move is next; 20-29=captures are sorted; 30-39=killers are sorted; 40-49=quiet moves are sorted; 50-59=all moves are processed
	unsigned int next_move;		// next move to be returnd. Start at 0, end at mc
	unsigned int moves_avalaible;// sorted
	unsigned int previos_stages_mc;
	unsigned int mc;			// unsorted
	unsigned int MCP_depth1;
	move sorted_moves[128];
	int score[128];				// move score. TT move is 1e6. Capture is 1000+. Killers are 0-100. Quiet moves are negative.
	unsigned char list[256];	// list of unsorted moves, up to 128 of them
	short int TTmove;
} move_list;

// function prototypes
unsigned int f_popcnt64(UINT64);
unsigned char f_BSF64(unsigned long*,unsigned __int64);
unsigned char f_BSR64(unsigned long*,unsigned __int64);
void init_material(void);
unsigned int get_all_moves(board*,unsigned char*);
unsigned int get_all_attack_moves(board*,unsigned char*);
unsigned int get_out_of_check_moves(board*,unsigned char*,unsigned int,unsigned int);
unsigned int find_all_get_out_of_check_moves_slow(board*,unsigned char*);
void make_null_move(board*);
void unmake_null_move(board*);
void make_move(board*,unsigned char,unsigned char,unmake*);
void unmake_move(board *, const unmake *);
unsigned int boards_are_the_same(board*,board*,unsigned char,unsigned char);
int Msearch(board*,const int,const unsigned int,int,int,unsigned int);
unsigned int get_piece_moves(board*,unsigned char*,unsigned char,unsigned int);
void init_moves(void);
void init_piece_square(void);
unsigned int cell_under_attack(board*,unsigned int,unsigned char);
unsigned int player_moved_into_check(board*,unsigned int,unsigned char);
unsigned int print_position(char*,board *);
UINT64 perft(board*,int);
unsigned int move_is_legal(board*,unsigned char,unsigned char);
uint64_t get_TT_hash_key(board*);
uint64_t get_pawn_hash_key(board*);
unsigned int get_mat_key(board *);
int get_piece_value(board *);
void get_scores(board*);
int pawn_score(board*);
unsigned int player_is_in_check(board*,unsigned int);
unsigned int move_list_is_good(board*,unsigned char*,unsigned int);
unsigned int checkmate(board*);
unsigned int bitboards_are_good(board*);
void init_board_FEN(const char*,board*);
void set_bitboards(board*);
void solve_prep(board*);
void init_all(unsigned int);
void init_board(unsigned int,board*);
void decode_move(unsigned char*,char*,board*);
unsigned int get_legal_moves(board*,unsigned char*);
void train(void);
void check_move(unsigned char*,unsigned char*,unsigned int);
int eval(board*);
int Qsearch(board*,unsigned int,int,int,int);
UINT64 attacks_bb_R(int,UINT64);
UINT64 attacks_bb_B(int,UINT64);
UINT64 flip_color(UINT64);
void init_threads(unsigned int);
unsigned int move_gives_check(board *,unsigned int,unsigned int);
unsigned int moving_piece_is_pinned(board *,unsigned int,unsigned int,unsigned int);
int f_timer(void);
void pass_message_to_GUI(const char*);
int16_t pass_forward_b(const board*);

// vars
extern FILE *f_log;
extern unsigned int HBITS;
extern unsigned int TTage;
extern int endgame_weight_all_i[];
extern unsigned char dist[64][64];
extern short int piece_square[6][2][64][2]; // [piece][player][square][midgame(0)/endgame(1)]
extern unsigned int depth0;

using ZobristArrayType = std::array<std::array<std::array<uint64_t, 64>, 2>, 6>;
extern const ZobristArrayType zorb;

extern int timeout;
extern int timeout_complete;
extern int time_start;
extern UINT64 *ph;
extern board b_m;
extern UINT64 ray_segment[64][64];
extern UINT64 knight_masks[];
extern UINT64 bishop_masks[];
extern UINT64 rook_masks[];
extern UINT64 king_masks[];
extern UINT64 dir_mask[5][64];

extern const uint8_t flips[64][8];
extern const UINT64 passed_mask[];
extern const UINT64 blocked_mask[];
extern unsigned char dir_norm[64][64];
extern const unsigned int mat_key_mult[];


extern const UINT64 pawn_attacks[2][64];
extern unsigned int tb_loaded,UseEGTBInsideSearch,EGTBProbeLimit;



#endif
