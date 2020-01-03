
#ifndef FIZBO_BITUTILS_H_INCLUDED
#define FIZBO_BITUTILS_H_INCLUDED

#include <cstdint>
#include <cstring>
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

	// not defined for m == 0
	static inline unsigned log2(uint32_t m)
	{
		return 31U - clz(m);
	}

	// not defined for m == 0
	static inline unsigned log2(uint64_t m)
	{
		return 63U - clz(m);
	}
};

struct TwinScore16 {
private:
	union {
		uint32_t u32;
		int16_t i16[2];
	};

public:
	TwinScore16() = default;

	inline TwinScore16(const int16_t (&x)[2])
	{
		memcpy(&u32, &x, 4);
	}

	inline TwinScore16(const int16_t sm, const int16_t se)
	{
		// we use this instead of i16 access to avoid false
		// dependencies, since most of the time we should be using the
		// 32-bit twin value
		u32 = static_cast<uint16_t>(sm) | (static_cast<uint32_t>(se) << 16);
	}

	explicit inline TwinScore16(const int16_t *tbl, size_t off)
	{
		memcpy(&u32, tbl + off, 4);
	}

	explicit inline TwinScore16(int32_t u)
	{
		u32 = u;
	}

	inline TwinScore16 &operator = (const int16_t (&x)[2])
	{
		memcpy(&u32, &x, 4);
		return *this;
	}
/*
	inline TwinScore16 &operator -= (const int16_t (&x)[2])
	{
		TwinScore16 other { x };

		u32 -= other.u32;
		return *this;
	}

	inline TwinScore16 &operator += (const int16_t (&x)[2])
	{
		TwinScore16 other { x };

		u32 += other.u32;
		return *this;
	}
*/
	inline TwinScore16 &operator -= (const TwinScore16 &x)
	{
		u32 -= x.u32;
		return *this;
	}

	inline TwinScore16 &operator += (const TwinScore16 &x)
	{
		u32 += x.u32;
		return *this;
	}

	inline TwinScore16 operator - () const
	{
		return TwinScore16(- getBoth());
	}

	inline uint32_t getBoth() const
	{
		return u32;
	}

	inline int16_t operator[] (size_t index) const
	{
		// we use this instead of i16 access to avoid false
		// dependencies, since most of the time we should be using the
		// 32-bit twin value
		return u32 >> (index == 0 ? 0 : 16);
	}

	inline int16_t &operator[] (size_t index)
	{
		// Allow modification of single score
		return i16[index];
	}

	inline void addFromTable(const int16_t *tbl, size_t off)
	{
		TwinScore16 other { tbl, off };

		u32 += other.u32;
	}

	inline void subFromTable(const int16_t *tbl, size_t off)
	{
		TwinScore16 other { tbl, off };

		u32 -= other.u32;
	}

	// note: weight = 0 (full midgame)..1024 (full endgame)
	int32_t blendByEndgameWeight(int32_t weight) const
	{
		const int32_t a { i16[0] };
		const int32_t b { i16[1] };

		// note: this rounds differently for positive/negative numbers:
		//   1025 >> 10   = 1 (round towards zero)
		//   1025 / 1024  = 1 (round towards zero)
		// But:
		//   -1025 >> 10  = -2 (round away from zero)
		//   -1025 / 1024 = -1 (round towards zero)
		return a + (((b - a) * weight) >> 10);
	}
};



#endif // FIZBO_BITUTILS_H_INCLUDED
