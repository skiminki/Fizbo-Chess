// pawn scores
#include "chess.h"
#include <intrin.h>
#include "coeffs.h"

UINT64* ph; // pawn hash table

const UINT64 passed_mask[64]={// applied to opposite color. To get passed pawns - forward, current and adjacent files. Zero out ranks 1, 7 and 8 - they never come into play.
	0,0x0000000000003F3F,0x0000000000001F1F,0x0000000000000F0F,0x0000000000000707,0x0000000000000303,0,0,
	0,0x00000000003F3F3F,0x00000000001F1F1F,0x00000000000F0F0F,0x0000000000070707,0x0000000000030303,0,0,
	0,0x000000003F3F3F00,0x000000001F1F1F00,0x000000000F0F0F00,0x0000000007070700,0x0000000003030300,0,0,
	0,0x0000003F3F3F0000,0x0000001F1F1F0000,0x0000000F0F0F0000,0x0000000707070000,0x0000000303030000,0,0,
	0,0x00003F3F3F000000,0x00001F1F1F000000,0x00000F0F0F000000,0x0000070707000000,0x0000030303000000,0,0,
	0,0x003F3F3F00000000,0x001F1F1F00000000,0x000F0F0F00000000,0x0007070700000000,0x0003030300000000,0,0,
	0,0x3F3F3F0000000000,0x1F1F1F0000000000,0x0F0F0F0000000000,0x0707070000000000,0x0303030000000000,0,0,
	0,0x3F3F000000000000,0x1F1F000000000000,0x0F0F000000000000,0x0707000000000000,0x0303000000000000,0,0};

const UINT64 blocked_mask[64]={// applied to the same color. To see if passed pawn is blocked my other player's pawn(s). Current file only.
	0x00000000000000fe,0x00000000000000fc,0x00000000000000f8,0x00000000000000f0,0x00000000000000e0,0x00000000000000c0,0x0000000000000080,0,
	0x000000000000fe00,0x000000000000fc00,0x000000000000f800,0x000000000000f000,0x000000000000e000,0x000000000000c000,0x0000000000008000,0,
	0x0000000000fe0000,0x0000000000fc0000,0x0000000000f80000,0x0000000000f00000,0x0000000000e00000,0x0000000000c00000,0x0000000000800000,0,
	0x00000000fe000000,0x00000000fc000000,0x00000000f8000000,0x00000000f0000000,0x00000000e0000000,0x00000000c0000000,0x0000000080000000,0,
	0x000000fe00000000,0x000000fc00000000,0x000000f800000000,0x000000f000000000,0x000000e000000000,0x000000c000000000,0x0000008000000000,0,
	0x0000fe0000000000,0x0000fc0000000000,0x0000f80000000000,0x0000f00000000000,0x0000e00000000000,0x0000c00000000000,0x0000800000000000,0,
	0x00fe000000000000,0x00fc000000000000,0x00f8000000000000,0x00f0000000000000,0x00e0000000000000,0x00c0000000000000,0x0080000000000000,0,
	0xfe00000000000000,0xfc00000000000000,0xf800000000000000,0xf000000000000000,0xe000000000000000,0xc000000000000000,0x8000000000000000,0};

static const UINT64 halfopen_mask[64]={// applied to opposite color. To get pawns on half-open file (used for candidate passed). Current file only.
	0x000000000000007f,0x000000000000003f,0x000000000000001f,0x000000000000000f,0x0000000000000007,0x0000000000000003,0x0000000000000001,0,
	0x0000000000007f00,0x0000000000003f00,0x0000000000001f00,0x0000000000000f00,0x0000000000000700,0x0000000000000300,0x0000000000000100,0,
	0x00000000007f0000,0x00000000003f0000,0x00000000001f0000,0x00000000000f0000,0x0000000000070000,0x0000000000030000,0x0000000000010000,0,
	0x000000007f000000,0x000000003f000000,0x000000001f000000,0x000000000f000000,0x0000000007000000,0x0000000003000000,0x0000000001000000,0,
	0x0000007f00000000,0x0000003f00000000,0x0000001f00000000,0x0000000f00000000,0x0000000700000000,0x0000000300000000,0x0000000100000000,0,
	0x00007f0000000000,0x00003f0000000000,0x00001f0000000000,0x00000f0000000000,0x0000070000000000,0x0000030000000000,0x0000010000000000,0,
	0x007f000000000000,0x003f000000000000,0x001f000000000000,0x000f000000000000,0x0007000000000000,0x0003000000000000,0x0001000000000000,0,
	0x7f00000000000000,0x3f00000000000000,0x1f00000000000000,0x0f00000000000000,0x0700000000000000,0x0300000000000000,0x0100000000000000,0};

static const UINT64 protected_mask[66]={// applied to the same color. To see if current pawn is protected. Need 2 trailing zeros.
	0x0000000000000000,0x0000000000000100,0x0000000000000200,0x0000000000000400,0x0000000000000800,0x0000000000001000,0x0000000000002000,0x0000000000004000,
	0x0000000000000000,0x0000000000010001,0x0000000000020002,0x0000000000040004,0x0000000000080008,0x0000000000100010,0x0000000000200020,0x0000000000400040,
	0x0000000000000000,0x0000000001000100,0x0000000002000200,0x0000000004000400,0x0000000008000800,0x0000000010001000,0x0000000020002000,0x0000000040004000,
	0x0000000000000000,0x0000000100010000,0x0000000200020000,0x0000000400040000,0x0000000800080000,0x0000001000100000,0x0000002000200000,0x0000004000400000,
	0x0000000000000000,0x0000010001000000,0x0000020002000000,0x0000040004000000,0x0000080008000000,0x0000100010000000,0x0000200020000000,0x0000400040000000,
	0x0000000000000000,0x0001000100000000,0x0002000200000000,0x0004000400000000,0x0008000800000000,0x0010001000000000,0x0020002000000000,0x0040004000000000,
	0x0000000000000000,0x0100010000000000,0x0200020000000000,0x0400040000000000,0x0800080000000000,0x1000100000000000,0x2000200000000000,0x4000400000000000,
	0x0000000000000000,0x0001000000000000,0x0002000000000000,0x0004000000000000,0x0008000000000000,0x0010000000000000,0x0020000000000000,0x0040000000000000,0,0};

static const UINT64 forward_protected_mask[64]={// applied to the same color. To see if current pawn and all its forward positions are protected.
	0x000000000000ff00,0x000000000000ff00,0x000000000000fe00,0x000000000000fc00,0x000000000000f800,0x000000000000f000,0x000000000000e000,0x000000000000c000,
	0x0000000000ff00ff,0x0000000000ff00ff,0x0000000000fe00fe,0x0000000000fc00fc,0x0000000000f800f8,0x0000000000f000f0,0x0000000000e000e0,0x0000000000c000c0,
	0x00000000ff00ff00,0x00000000ff00ff00,0x00000000fe00fe00,0x00000000fc00fc00,0x00000000f800f800,0x00000000f000f000,0x00000000e000e000,0x00000000c000c000,
	0x000000ff00ff0000,0x000000ff00ff0000,0x000000fe00fe0000,0x000000fc00fc0000,0x000000f800f80000,0x000000f000f00000,0x000000e000e00000,0x000000c000c00000,
	0x0000ff00ff000000,0x0000ff00ff000000,0x0000fe00fe000000,0x0000fc00fc000000,0x0000f800f8000000,0x0000f000f0000000,0x0000e000e0000000,0x0000c000c0000000,
	0x00ff00ff00000000,0x00ff00ff00000000,0x00fe00fe00000000,0x00fc00fc00000000,0x00f800f800000000,0x00f000f000000000,0x00e000e000000000,0x00c000c000000000,
	0xff00ff0000000000,0xff00ff0000000000,0xfe00fe0000000000,0xfc00fc0000000000,0xf800f80000000000,0xf000f00000000000,0xe000e00000000000,0xc000c00000000000,
	0x00ff000000000000,0x00ff000000000000,0x00fe000000000000,0x00fc000000000000,0x00f8000000000000,0x00f0000000000000,0x00e0000000000000,0x00c0000000000000};

static const UINT64 is_mask[8]={0xff00,0xff00ff,0xff00ff00,0xff00ff0000,0xff00ff000000,0xff00ff00000000,0xff00ff0000000000,0xff000000000000};// isolated pawn mask for files 1-8

// takes bit*64+bit2, returns index 0-575.
unsigned short int index_pawn[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4,0,0,5,6,7,8,9,10,0,0,11,12,13,14,15,16,0,0,17,18,19,20,21,22,0,0,23,24,25,26,27,28,0,0,29,30,31,32,33,34,0,0,35,36,37,38,39,40,0,0,41,42,43,44,45,46,0,0,0,0,47,48,49,50,0,0,51,52,53,54,55,56,0,0,57,58,59,60,61,62,0,0,63,64,65,66,67,68,0,0,69,70,71,72,73,74,0,0,75,76,77,78,79,80,0,0,81,82,83,84,85,86,0,0,42,87,88,89,90,91,0,0,0,0,0,92,93,94,0,
0,95,96,97,98,99,100,0,0,101,102,103,104,105,106,0,0,107,108,109,110,111,112,0,0,113,114,115,116,117,118,0,0,119,120,121,122,123,124,0,0,125,126,127,128,129,130,0,0,43,88,131,132,133,134,0,0,0,0,0,0,135,136,0,0,137,138,139,140,141,142,0,0,143,144,145,146,147,148,0,0,149,150,151,152,153,154,0,0,155,156,157,158,159,160,0,0,161,162,163,164,165,166,0,0,167,168,169,170,171,172,0,0,44,89,132,173,174,175,0,0,0,0,0,0,0,176,0,0,177,178,179,180,181,182,0,0,183,184,185,186,187,188,0,0,189,190,191,192,193,194,0,0,195,196,197,198,199,200,0,0,201,202,203,204,205,206,0,0,207,208,209,210,211,212,0,0,45,90,133,174,213,214,0,0,0,0,0,0,0,0,0,0,215,216,217,218,219,220,0,
0,221,222,223,224,225,226,0,0,227,228,229,230,231,232,0,0,233,234,235,236,237,238,0,0,239,240,241,242,243,244,0,0,245,246,247,248,249,250,0,0,46,91,134,175,214,251,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,252,253,254,255,256,0,0,257,258,259,260,261,262,0,
0,263,264,265,266,267,268,0,0,269,270,271,272,273,274,0,0,275,276,277,278,279,280,0,0,281,282,283,284,285,286,0,0,35,81,125,167,207,245,0,0,0,0,0,0,0,0,0,0,0,0,287,288,289,290,0,0,291,292,293,294,295,296,0,0,297,298,299,300,301,302,0,0,303,304,305,306,307,308,0,0,309,310,311,312,313,314,0,0,282,315,316,317,318,319,0,0,36,82,126,168,208,246,0,0,0,0,0,0,0,0,0,0,0,0,0,320,321,322,0,0,323,324,325,326,327,328,0,0,329,330,331,332,333,334,0,0,335,336,337,338,339,340,0,0,341,342,343,344,345,346,0,0,283,316,347,348,349,350,0,0,37,83,127,169,209,247,0,0,0,0,0,0,0,0,0,0,0,0,0,0,351,352,0,0,353,354,355,356,357,358,0,0,359,360,361,362,363,364,0,
0,365,366,367,368,369,370,0,0,371,372,373,374,375,376,0,0,284,317,348,377,378,379,0,0,38,84,128,170,210,248,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,380,0,0,381,382,383,384,385,386,0,0,387,388,389,390,391,392,0,0,393,394,395,396,397,398,0,0,399,400,401,402,403,404,0,0,285,318,349,378,405,406,0,0,39,85,129,171,211,249,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,407,408,409,410,411,412,0,0,413,414,415,416,417,418,0,0,419,420,421,422,423,424,0,0,425,426,427,428,429,430,0,0,286,319,350,379,406,431,0,0,40,86,130,172,212,250,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,432,433,434,435,436,0,0,437,438,439,440,441,442,0,0,443,444,445,446,447,448,0,0,449,450,451,452,453,454,0,0,275,309,341,371,399,425,0,0,29,75,119,161,201,239,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,455,456,457,458,0,0,459,460,461,462,463,464,0,0,465,466,467,468,469,470,0,0,450,471,472,473,474,475,0,
0,276,310,342,372,400,426,0,0,30,76,120,162,202,240,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,476,477,478,0,0,479,480,481,482,483,484,0,0,485,486,487,488,489,490,0,0,451,472,491,492,493,494,0,0,277,311,343,373,401,427,0,0,31,77,121,163,203,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,495,496,0,0,497,498,499,500,501,502,0,0,503,504,505,506,507,508,0,0,452,473,492,509,510,511,0,0,278,312,344,374,402,428,0,0,32,78,122,164,204,242,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,512,0,0,513,514,515,516,517,518,0,0,519,520,521,522,523,524,0,0,453,474,493,510,525,526,0,0,279,313,345,375,403,429,0,
0,33,79,123,165,205,243,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,527,528,529,530,531,532,0,0,533,534,535,536,537,538,0,0,454,475,494,511,526,539,0,0,280,314,346,376,404,430,0,0,34,80,124,166,206,244,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,540,541,542,543,544,0,0,545,546,547,548,549,550,0,0,443,465,485,503,519,533,0,0,269,303,335,365,393,419,0,0,23,69,113,155,195,233,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,551,552,553,554,0,0,546,555,556,557,558,559,0,0,444,466,486,504,520,534,0,0,270,304,336,366,394,420,0,0,24,70,114,156,196,234,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,560,561,562,0,0,547,556,563,564,565,566,0,0,445,467,487,505,521,535,0,0,271,305,337,367,395,421,0,0,25,71,115,157,197,235,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,567,568,0,0,548,557,564,569,570,571,0,0,446,468,488,506,522,536,0,0,272,306,338,368,396,422,0,0,26,72,116,158,198,236,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,572,0,0,549,558,565,570,573,574,0,0,447,469,489,507,523,537,0,0,273,307,339,369,397,423,0,0,27,73,117,159,199,237,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,550,559,566,571,574,575,0,0,448,470,490,508,524,538,0,0,274,308,340,370,398,424,0,0,28,74,118,160,200,238,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,540,541,542,543,544,0,0,437,459,479,497,513,527,0,0,263,297,329,359,387,413,0,0,17,63,107,149,189,227,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,551,552,553,554,0,0,438,460,480,498,514,528,0,0,264,298,330,360,388,414,0,0,18,64,108,150,190,228,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,560,561,562,0,0,439,461,481,499,515,529,0,0,265,299,331,361,389,415,0,0,19,65,109,151,191,229,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,567,568,0,0,440,462,482,500,516,530,0,0,266,300,332,362,390,416,0,0,20,66,110,152,192,230,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,572,0,0,441,463,483,501,517,531,0,0,267,301,333,363,391,417,0,0,21,67,111,153,193,231,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,442,464,484,502,518,532,0,0,268,302,334,364,392,418,0,0,22,68,112,154,194,232,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,432,433,434,435,436,0,0,257,291,323,353,381,407,0,0,11,57,101,143,183,221,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,455,456,457,458,0,0,258,292,324,354,382,408,0,0,12,58,102,144,184,222,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,476,477,478,0,
0,259,293,325,355,383,409,0,0,13,59,103,145,185,223,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,495,496,0,0,260,294,326,356,384,410,0,0,14,60,104,146,186,224,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,512,0,0,261,295,327,357,385,411,0,0,15,61,105,147,187,225,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,262,296,328,358,386,412,0,
0,16,62,106,148,188,226,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,252,253,254,255,256,0,0,5,51,95,137,177,215,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,287,288,289,290,0,0,6,52,96,138,178,216,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,320,321,322,0,0,7,53,97,139,179,217,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,351,352,0,0,8,54,98,140,180,218,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,380,0,0,9,55,99,141,181,219,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,56,100,142,182,220,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,47,48,49,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,93,94,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,135,136,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,176};



// Exchange the ranks 1-8, 2-7, 3-6, 4-5. Flips point of view between white and black.
UINT64 flip_color(UINT64 b){
	b=((b>>1)&0x5555555555555555)|((b<<1)&0xAAAAAAAAAAAAAAAA);
	b=((b>>2)&0x3333333333333333)|((b<<2)&0xCCCCCCCCCCCCCCCC);
	b=((b>>4)&0x0F0F0F0F0F0F0F0F)|((b<<4)&0xF0F0F0F0F0F0F0F0);
	return b;
}

#if TRAIN
extern _declspec(thread) int pawn_deriv_coeffs[320];
extern _declspec(thread) unsigned int use_hash;
#endif

typedef struct{
	unsigned int lock;
	int sk4;
} pTT; // 8 bytes

// deriv coeffs:
// 1=isolated
// 2-7=passed, on ranks 2-7 (6 of them)
// 8=passed protected
// 9-22=
// 23-28=candidate passed, on ranks 2-7 (6 of them)
// 29-47=
// 48=backward
// 49-85=protected/protecting/attacked/mobile=3*6*2=6*6=36.
// 86: free
//void pass_forward_b(board*,short int*);
int pawn_score(board *b){// return scores for WHITE.
	volatile UINT64 *h1;
	UINT64 w_bb,b_bb,w_bb_t,b_bb_t;
	pTT h;
	unsigned long bit;
	unsigned int i,j,sq[2][8],sq_cnt[2]; //sq: list of all pawns.
	int sk4;
	
	// look in pawn hash table
	h1=&ph[b->pawn_hash_key%PHSIZE];
	h=((pTT*)h1)[0]; // atomic read
	#if TRAIN
	#else
	if( h.lock==((unsigned int*)&b->pawn_hash_key)[1] )
		return(h.sk4);
	#endif
	h.lock=((unsigned int*)&b->pawn_hash_key)[1]; // save lock now (into h, not into real memory!)
	sk4=0;// init scores	

	// populate bitboards
	w_bb=b->piececolorBB[0][0];//white pawn
	b_bb=flip_color(b->piececolorBB[0][1]);//black pawn, flipped so that it is from point of view of white
	
	// loop over all pawns
	w_bb_t=w_bb;
	sq_cnt[0]=0;
	while( w_bb_t ){// white
		GET_BIT(w_bb_t)
		sq[0][sq_cnt[0]++]=bit; // record position of this pawn

		#if calc_pst==1		
		sk4+=((int*)&piece_square[0][0][bit][0])[0];// white P PST
		#endif

		// count isolated pawns. Here i count doubled isolated pawns as 2 isolated.
		if( (is_mask[bit>>3]&w_bb)==0 ){
			sk4-=((int*)&adj[O_ISOLATED])[0];
			#if TRAIN
			pawn_deriv_coeffs[1]--; // 1=isolated************************************************************************************************************
			#endif
		}

		// count passed and candidate passed
		i=(unsigned int)popcnt64l(w_bb&protected_mask[bit]);// protected count, 0-2
		if( !((w_bb&blocked_mask[bit])|(b_bb&halfopen_mask[bit])) ){// not blocked by anyone - half-open file. Could be (candidate) passed
			if( (b_bb&passed_mask[bit])==0 ){ // passed pawn
				if( (bit&7)<6 )// skip rank 7
					sk4+=((int*)&adj[O_PASSED+(bit&7)*2-2])[0];
				#if TRAIN
				pawn_deriv_coeffs[2+(bit&7)-1]++; // 2-7=passed, on ranks 2-7************************************************************************************************************
				#endif
				if( i ){// protected passed pawn
					sk4+=((int*)&adj[O_PASSED_PROTECTED])[0];
					#if TRAIN
					pawn_deriv_coeffs[8]++; // 8=passed protected************************************************************************************************************
					#endif
				}
			}else if( popcnt64l(w_bb&forward_protected_mask[bit])>=popcnt64l(b_bb&(passed_mask[bit]^halfopen_mask[bit])) ){// assume candidate if current and forward squares have same of more defenders as attackers
				if( (bit&7)<6 )// skip rank 7
					sk4+=((int*)&adj[O_CAND_PASSED+(bit&7)*2-2])[0];
				#if TRAIN
				pawn_deriv_coeffs[23+(bit&7)-1]++; // 23-28=candidate passed, on ranks 2-7************************************************************************************************************
				#endif
			}
		}
	}
	b_bb_t=b_bb;
	sq_cnt[1]=0;
	while( b_bb_t ){// black
		GET_BIT(b_bb_t)
		sq[1][sq_cnt[1]++]=bit; // record position of this pawn - in flipped format(white's point of view)

		#if calc_pst==1
		sk4+=((int*)&piece_square[0][1][flips[bit][1]][0])[0];// black P PST
		#endif

		// count isolated pawns. Here i count doubled isolated pawns as 2 isolated.
		if( (is_mask[bit>>3]&b_bb)==0 ){
			sk4+=((int*)&adj[O_ISOLATED])[0];
			#if TRAIN
			pawn_deriv_coeffs[1]++; // 1=isolated************************************************************************************************************
			#endif
		}

		// count passed and candidate passed
		i=(unsigned int)popcnt64l(b_bb&protected_mask[bit]);
		if( !((b_bb&blocked_mask[bit])|(w_bb&halfopen_mask[bit])) ){// not blocked by anyone - half-open file. Could be (candidate) passed
			if( (w_bb&passed_mask[bit])==0 ){ // passed pawn
				if( (bit&7)<6 )// skip rank 7
					sk4-=((int*)&adj[O_PASSED+(bit&7)*2-2])[0];
				#if TRAIN
				pawn_deriv_coeffs[2+(bit&7)-1]--; // 2-7=passed, on ranks 2-7************************************************************************************************************
				#endif
				if( i ){// protected passed pawn
					sk4-=((int*)&adj[O_PASSED_PROTECTED])[0];
					#if TRAIN
					pawn_deriv_coeffs[8]--; // 8=passed protected************************************************************************************************************
					#endif
				}
			}else if( popcnt64l(b_bb&forward_protected_mask[bit])>=popcnt64l(w_bb&(passed_mask[bit]^halfopen_mask[bit])) ){// assume candidate if current and forward squares have same of more defenders as attackers
				if( (bit&7)<6 )// skip rank 7
					sk4-=((int*)&adj[O_CAND_PASSED+(bit&7)*2-2])[0];
				#if TRAIN
				pawn_deriv_coeffs[23+(bit&7)-1]--; // 23-28=candidate passed, on ranks 2-7************************************************************************************************************
				#endif
			}
		}
	}

	// my pawn vs my pawn - new index development logic, with look-up table.
	for(i=0;i+1<sq_cnt[0];++i){
		unsigned int bit=sq[0][i];
		for(j=i+1;j<sq_cnt[0];++j){// loop over P2, P2>P1
			unsigned int index=index_pawn[bit*64+sq[0][j]];
			sk4+=((int*)&adj[O_PP+index*2])[0];
		}
	}
	for(i=0;i+1<sq_cnt[1];++i){// all bits are already from white's POV
		unsigned int bit=sq[1][i];
		for(j=i+1;j<sq_cnt[1];++j){// loop over P2, P2>P1
			unsigned int index=index_pawn[bit*64+sq[1][j]];
			sk4-=((int*)&adj[O_PP+index*2])[0];
		}
	}
	
	// add score to pawn hash table. Always replace.
	#if TRAIN
	#else
	h.sk4=sk4;
	((pTT*)h1)[0]=h;		// atomic write
	#endif

	// return score for WHITE.
	return(sk4);
}