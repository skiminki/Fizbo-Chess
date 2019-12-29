
#include "chess.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>

#include "vectorutils.h"

uint32_t checks = 0;
uint32_t failures = 0;


static void testCondition(const char *file, int line, const char *cond, bool value)
{
	++checks;
	if (!value) {
		printf("%s:%d Condition '%s' failed\n", file, line, cond);
		++failures;
	}
}

#define VERIFY(cond) testCondition(__FILE__, __LINE__, #cond, cond)

int16_t inspect_sumOfElements16i(__m128i v0)
{
	return VectorUtils::sumOfElements16i(v0);
}

#ifdef __AVX__
int16_t inspect_sumOfElements16i(__m256i v0)
{
	return VectorUtils::sumOfElements16i(v0);
}
#endif

uint32_t inspect_blsr(uint32_t a)
{
	return BitUtils::blsr(a);
}

uint64_t inspect_blsr(uint64_t a)
{
	return BitUtils::blsr(a);
}


int main(int argc, char **argv)
{
	// BSF/BSR
	{
		uint32_t bitScanIndex { };

		VERIFY(BSF64l(&bitScanIndex, uint64_t { 12 }) != 0);
		VERIFY(bitScanIndex == 2);

		VERIFY(BSR64l(&bitScanIndex, uint64_t { 12 }) != 0);
		VERIFY(bitScanIndex == 3);
	}

	// Vector utils
	{
		__m128i v0 = _mm_set_epi16(19, 17, 13, 11, 7, 5, 3, 2 );

		int16_t sum { VectorUtils::sumOfElements16i(v0) };
		VERIFY(sum == (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19));
	}
#if __AVX__
	// Vector utils
	{

		__m256i v0 =
			_mm256_set_epi16(
				53, 47, 43, 41, 37, 31, 29, 23,
				19, 17, 13, 11, 7,  5,  3,  2 );

		int16_t sum { VectorUtils::sumOfElements16i(v0) };
		VERIFY(sum == (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29 + 31 + 37 + 41 + 43 + 47 + 53));
	}
#endif
	printf("Done %u checks, %u failures\n", checks, failures);

	return failures == 0 ? 0 : 1;
}
