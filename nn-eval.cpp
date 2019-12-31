
#include "chess.h"
#include "nn-weights.h"
#include "vectorutils.h"

#if USE_AVX

// compute output of network, taking board as input. Using AVX2 instructions.
int16_t pass_forward_b(const board *b)
{
	constexpr __m256i vzero { };
	__m256i v0 { };
	__m256i v1 { };
	__m256i v2 { };
	__m256i v3 { };

	uint64_t m { b->colorBB[0] | b->colorBB[1] }; // material bits

	while (m != 0) {
		uint32_t bit;
		GET_BIT(m)

		const unsigned int q = b->piece[bit];		// unformatted piece
		const unsigned int p = (q&7)-1;				// 0-6
		const unsigned int k = (p+6*(q>>7))+bit*12-1;// index

		const __m256i *weightsLine = (const __m256i *)&NnWeights::c1iaProd[k][0];

		v0 = _mm256_add_epi16(v0, weightsLine[0]);
		v1 = _mm256_add_epi16(v1, weightsLine[1]);
		v2 = _mm256_add_epi16(v2, weightsLine[2]);
		v3 = _mm256_add_epi16(v3, weightsLine[3]);
	}

	 // add/sub: assume first 32 are +, last 32 are -.
	const __m256i total =
		_mm256_sub_epi16(_mm256_add_epi16(_mm256_max_epi16(v0, vzero), _mm256_max_epi16(v1, vzero)),
				 _mm256_add_epi16(_mm256_max_epi16(v2, vzero), _mm256_max_epi16(v3, vzero)));

	return VectorUtils::sumOfElements16i(total);
}

#else

// compute output of network, taking board as input. Using SSE2 instructions.
int16_t pass_forward_b(const board *b)
{
	constexpr __m128i vzero { };
	__m128i v0 { };
	__m128i v1 { };
	__m128i v2 { };
	__m128i v3 { };
	__m128i v4 { };
	__m128i v5 { };
	__m128i v6 { };
	__m128i v7 { };

	uint64_t m { b->colorBB[0] | b->colorBB[1] }; // material bits

	while (m != 0) {
		uint32_t bit;
		GET_BIT(m)

		const unsigned int q = b->piece[bit];		// unformatted piece
		const unsigned int p = (q&7)-1;			// 0-6
		const unsigned int k = (p+6*(q>>7))+bit*12-1;	// index

		const __m128i *weightsLine = (const __m128i *)&NnWeights::c1iaProd[k][0];

		v0 = _mm_add_epi16(v0, weightsLine[0]);
		v1 = _mm_add_epi16(v1, weightsLine[1]);
		v2 = _mm_add_epi16(v2, weightsLine[2]);
		v3 = _mm_add_epi16(v3, weightsLine[3]);
		v4 = _mm_add_epi16(v4, weightsLine[4]);
		v5 = _mm_add_epi16(v5, weightsLine[5]);
		v6 = _mm_add_epi16(v6, weightsLine[6]);
		v7 = _mm_add_epi16(v7, weightsLine[7]);
	}

	// add/sub: assume first 32 are +, last 32 are -.
	const __m128i total =
		_mm_sub_epi16(_mm_add_epi16(_mm_add_epi16(_mm_max_epi16(v0,vzero), _mm_max_epi16(v1,vzero)),
					    _mm_add_epi16(_mm_max_epi16(v2,vzero), _mm_max_epi16(v3,vzero))),
			      _mm_add_epi16(_mm_add_epi16(_mm_max_epi16(v4,vzero), _mm_max_epi16(v5,vzero)),
					    _mm_add_epi16(_mm_max_epi16(v6,vzero), _mm_max_epi16(v7,vzero))));

	return VectorUtils::sumOfElements16i(total);	// return score for white
}

#endif
