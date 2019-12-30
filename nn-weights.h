#ifndef FIZBO_NN_WEIGHTS_H_INCLUDED
#define FIZBO_NN_WEIGHTS_H_INCLUDED

#include <cstddef>
#include <cstdint>

struct NnWeights
{
	static constexpr size_t numInputs { 769 }; // 12 piece types on 64 squares, plus bias: 64*12+1. Bias goes in first.
	static constexpr size_t numN1 { 64 }; 	// number of first layer neurons. Plus bias.

	// the weight data
	static const int16_t c1iaProd[numInputs - 2][numN1];
};

#endif // FIZBO_NN_WEIGHTS_H_INCLUDED

