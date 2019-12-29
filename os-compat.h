#ifndef FIZBO_OS_COMPAT_H_INCLUDED
#define FIZBO_OS_COMPAT_H_INCLUDED

#include <atomic>

#include "bitutils.h"

#define NOINLINE __attribute__((noinline))



inline uint8_t BitScanForward(uint32_t *index, uint32_t mask)
{
	if (mask) {
		*index = BitUtils::ctz(mask);
		return 1;
	} else {
		*index = 0;
		return 0;
	}
}

inline uint8_t BitScanForward64(uint32_t *index, uint64_t mask)
{
	if (mask) {
		*index = BitUtils::ctz(mask);
		return 1;
	} else {
		*index = 0;
		return 0;
	}
}

inline uint8_t BitScanReverse(uint32_t *index, uint32_t mask)
{
	if (mask) {
		*index = 31 - BitUtils::clz(mask);
		return 1;
	} else {
		*index = 0;
		return 0;
	}
}

inline uint8_t BitScanReverse64(uint32_t *index, uint64_t mask)
{
	if (mask) {
		*index = 63 - BitUtils::clz(mask);
		return 1;
	} else {
		*index = 0;
		return 0;
	}
}


inline void data_prefetch(const void *addr)
{
	__builtin_prefetch(addr);
}

inline void remove_InterlockedAnd(volatile int32_t *dest, int32_t value)
{
	std::atomic_fetch_and_explicit((std::atomic_int32_t *)dest, value, std::memory_order_relaxed);
}

inline void InterlockedAnd64(volatile int64_t *dest, int64_t value)
{
	std::atomic_fetch_and_explicit((std::atomic_int64_t *)dest, value, std::memory_order_relaxed);
}

inline void InterlockedAnd64(volatile uint64_t *dest, uint64_t value)
{
	std::atomic_fetch_and_explicit((std::atomic_uint64_t *)dest, value, std::memory_order_relaxed);
}

inline void InterlockedOr64(volatile int64_t *dest, int64_t value)
{
	std::atomic_fetch_or_explicit((std::atomic_int64_t *)dest, value, std::memory_order_relaxed);
}

inline void InterlockedOr64(volatile uint64_t *dest, uint64_t value)
{
	std::atomic_fetch_or_explicit((std::atomic_uint64_t *)dest, value, std::memory_order_relaxed);
}

inline void InterlockedExchangeAdd64(volatile int64_t *dest, int64_t value)
{
	std::atomic_fetch_add_explicit((std::atomic_int64_t *)dest, value, std::memory_order_relaxed);
}

inline void InterlockedExchangeAdd64(volatile uint64_t *dest, uint64_t value)
{
	std::atomic_fetch_add_explicit((std::atomic_uint64_t *)dest, value, std::memory_order_relaxed);
}

uint32_t getTimeMs();
void sleepMs(uint32_t ms);

#endif
