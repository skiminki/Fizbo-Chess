
#ifndef FIZBO_BITUTILS_H_INCLUDED
#define FIZBO_BITUTILS_H_INCLUDED

#include <cstdint>
#include <immintrin.h>

struct BitUtils {

	static inline unsigned ctz(unsigned m)
	{
		return __builtin_ctz(m);
	}

	static inline unsigned ctz(unsigned long m)
	{
		return __builtin_ctzl(m);
	}

	static inline unsigned ctz(unsigned long long m)
	{
		return __builtin_ctzll(m);
	}

	static inline unsigned clz(unsigned m)
	{
		return __builtin_clz(m);
	}

	static inline unsigned clz(unsigned long m)
	{
		return __builtin_clzl(m);
	}

	static inline unsigned clz(unsigned long long m)
	{
		return __builtin_clzll(m);
	}

	static inline unsigned popcount(unsigned m)
	{
		return __builtin_popcount(m);
	}

	static inline unsigned popcount(unsigned long m)
	{
		return __builtin_popcountl(m);
	}

	static inline unsigned popcount(unsigned long long m)
	{
		return __builtin_popcountll(m);
	}

	static inline uint32_t blsr(uint32_t x)
	{
#ifdef __BMI__
		// Note: this is technically not required, since GCC/CLANG
		// should produce BLSR anyways from the below
		// expression. However, let's be extra sure.
		return _blsr_u32(x);
#else
		return x & (x - 1);
#endif
	}

	static inline uint64_t blsr(uint64_t x)
	{
#ifdef __BMI__
		// Note: this is technically not required, since GCC/CLANG
		// should produce BLSR anyways from the below
		// expression. However, let's be extra sure.
		return _blsr_u64(x);
#else
		return x & (x - 1);
#endif
	}
};

#endif // FIZBO_BITUTILS_H_INCLUDED
