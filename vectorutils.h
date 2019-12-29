
#ifndef FIZBO_VECTORUTILS_H_INCLUDED
#define FIZBO_VECTORUTILS_H_INCLUDED

#include <immintrin.h>

struct VectorUtils {
/*
	static void printEpi16(__m128i v0)
	{
		printf("vec: %3" PRId16 " %3" PRId16 " %3" PRId16 " %3" PRId16 " %3" PRId16 " %3" PRId16 " %3" PRId16 " %3" PRId16 "\n",
		       _mm_extract_epi16(v0, 0),
		       _mm_extract_epi16(v0, 1),
		       _mm_extract_epi16(v0, 2),
		       _mm_extract_epi16(v0, 3),
		       _mm_extract_epi16(v0, 4),
		       _mm_extract_epi16(v0, 5),
		       _mm_extract_epi16(v0, 6),
		       _mm_extract_epi16(v0, 7));
	}

#ifdef __AVX__
	static void printEpi16(__m256i v0)
	{
		__m128i t0 = _mm256_castsi256_si128(v0);
		__m128i t1 = _mm256_extracti128_si256(v0, 1);

		printf("vec: %3" PRId16 " %3" PRId16 " %3" PRId16 " %3" PRId16 " %3" PRId16 " %3" PRId16 " %3" PRId16 " %3" PRId16
		       " %3" PRId16 " %3" PRId16 " %3" PRId16 " %3" PRId16 " %3" PRId16 " %3" PRId16 " %3" PRId16 " %3" PRId16 "\n",
		       _mm_extract_epi16(t0, 0),
		       _mm_extract_epi16(t0, 1),
		       _mm_extract_epi16(t0, 2),
		       _mm_extract_epi16(t0, 3),
		       _mm_extract_epi16(t0, 4),
		       _mm_extract_epi16(t0, 5),
		       _mm_extract_epi16(t0, 6),
		       _mm_extract_epi16(t0, 7),
		       _mm_extract_epi16(t1, 0),
		       _mm_extract_epi16(t1, 1),
		       _mm_extract_epi16(t1, 2),
		       _mm_extract_epi16(t1, 3),
		       _mm_extract_epi16(t1, 4),
		       _mm_extract_epi16(t1, 5),
		       _mm_extract_epi16(t1, 6),
		       _mm_extract_epi16(t1, 7));
	}
#endif
*/

	static inline int16_t sumOfElements16i(__m128i v0)
	{
		__m128i tmp, tmp2;

		tmp2 = _mm_shuffle_epi32(v0, _MM_SHUFFLE(1, 0, 3, 2));

		tmp = _mm_add_epi16(v0, tmp2); // sum 64
		tmp2 = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(2, 3, 0, 1));

		tmp = _mm_add_epi16(tmp, tmp2); // sum 32

		tmp2 = _mm_shufflelo_epi16(tmp, _MM_SHUFFLE(2, 3, 0, 1));
		tmp = _mm_add_epi16(tmp, tmp2); // sum 16

		return _mm_cvtsi128_si64(tmp); // faster than  _mm_extract_epi16(tmp, 0);
	}

#ifdef __AVX__
	static inline int16_t sumOfElements16i(__m256i v0)
	{
		__m128i t0, t1;

		t0 = _mm256_castsi256_si128(v0);
		t1 = _mm256_extracti128_si256(v0, 1);

		t0 = _mm_add_epi16(t0, t1);

		return sumOfElements16i(t0);
	}
#endif

};


#endif
