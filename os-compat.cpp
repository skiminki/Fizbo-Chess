#include "os-compat.h"

#include <ctime>

uint32_t getTimeMs();
void sleepMs(uint32_t ms);

uint32_t getTimeMs()
{
	constexpr uint64_t nsecsPerSec { 1000000000 };
	constexpr uint64_t nsecsPerMsec { 1000000 };

	timespec tp { };
	clock_gettime(CLOCK_MONOTONIC, &tp);

	uint64_t nsecs = tp.tv_nsec;
	nsecs += tp.tv_sec * nsecsPerSec;

	return nsecs / nsecsPerMsec;
}

void sleepMs(uint32_t ms)
{
	constexpr uint64_t nsecsPerSec { 1000000000 };
	constexpr uint64_t nsecsPerMsec { 1000000 };

	const uint64_t nsecs = ms * nsecsPerMsec;

	timespec tp { };

	tp.tv_sec = nsecs / nsecsPerSec;
	tp.tv_nsec = nsecs % nsecsPerSec;

	nanosleep(&tp, NULL);
}
