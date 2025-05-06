package mcmc

import (
	"math/rand/v2"
)

// Sample uses the buffer to generate a sample from the Metropolis Hastings algorithm.
//
// The resulting sample and score can be accessed using [LastSample] and [LastScore].
//
// scoreFn is a sample scoring function with the property F(X) = scoreFn(X) / NC,
// where F is the target distribution and NC is the possibly difficult to calculate
// normalizing constant.
func (b *Buffer) Sample(scoreFn func([]float64) float64) {
	r := b.Rand
	// x_t = x_0
	xt := b.Buf[:b.SampleDims]
	scoreT := b.LastScore
	xp := b.Buf[b.SampleDims : 2*b.SampleDims]
	theta := b.Theta

	// Generate a candidate x' from proposal distribution g(x|y).
	proposal(r, xp, xt, theta)
	// Calculate the acceptance probability a = min(1, [P(x') g(x'|x_t)] / [P(x_t) g(x_t|x')]).
	// Optimizations:
	// 	* Assuming symmetric proposal distribution, g(x'|x_t) = g(x_t|x'), so we leave those out.
	// 	* Avoid generating u ~ U(0,1) except when needed.
	// 	* Avoid division in acceptance term in some cases.
	// Handle special cases such as negative scores, zero, NaN.
	switch scoreP := scoreFn(xp); {
	case scoreP == 0: // Special case rejection:
	case scoreT == 0, scoreP >= scoreT, (scoreP < 0) == (scoreT < 0) && r.Float64() < scoreP/scoreT: // Accept:
		copy(xt, xp) // Copy sample st. x = x'
		b.LastScore = scoreP
	}
	// Otherwise, Reject x' and set x_{t+1} = x_t (nop).
}

func proposal(r *rand.Rand, dst, src, theta []float64) {
	for i := range dst {
		dst[i] = src[i] + r.NormFloat64()*theta[i]
	}
}
