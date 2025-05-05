package mcmc

import (
	"iter"
	"math"
	"math/rand/v2"
	"slices"
)

// Settings applied to [Sample].
type Settings struct {
	RandSrc rand.Source
	X0      []float64
	Theta   []float64 // stddev for proposal distribution
}

// Sample returns an iterator which emits samples from the Metropolis Hastings algorithm.
//
// scoreFn is a sample scoring function with the property F(X) = scoreFn(X) / NC,
// where F is the target distribution and NC is the possibly difficult to calculate
// normalizing constant.
func Sample(scoreFn func(x []float64) float64, opts Settings) iter.Seq[[]float64] {
	return func(yield func([]float64) bool) {
		r := rand.New(opts.RandSrc)
		// x_t = x_0
		xt := slices.Clone(opts.X0)
		xp := make([]float64, len(opts.X0))
		scoret := scoreFn(xt)
		theta := opts.Theta
		// t = 0
		// Yield x_t
		for yield(xt) {
			// Generate a candidate x' from proposal distribution g(x|y).
			proposal(r, xp, xt, theta)
			// Calculate the acceptance probability a = min(1, [P(x') g(x'|x_t)] / [P(x_t) g(x_t|x')]).
			//  Note that with a symmetric proposal distribution: g(x'|x_t) = g(x_t|x') so we can leave those out.
			// Avoid division in acceptance term in some cases.
			// Handle NaN and division by 0.
			// Only generate u ~ U(0,1) when needed.
			// If u < a, then Accept x' and set x_{t+1} = x'.
			if scorep := scoreFn(xp); scorep >= scoret || scoret == 0 || math.IsNaN(scorep) || math.IsNaN(scoret) || r.Float64() < scorep/scoret {
				copy(xt, xp)
				scoret = scorep
			}
			// Otherwise, Reject x' and set x_{t+1} = x_t (nop).
		}
	}
}

func proposal(r *rand.Rand, dst, src, theta []float64) {
	for i := range dst {
		dst[i] = src[i] + r.NormFloat64()*theta[i]
	}
}
