package mcmc

import (
	"math"
	"math/rand/v2"
	"testing"
)

func scoreFunc(x []float64) float64 {
	const (
		mu    = 0
		theta = 1
	)
	var score float64
	for _, e := range x {
		score += math.Exp(-math.Pow(e-mu, 2) / (2 * theta * theta))
	}
	return score
}

func TestSample(t *testing.T) {
	buf := Buffer{
		SampleDims: 1,
		Buf:        []float64{40, 0},
		Theta:      []float64{1},
		Rand:       rand.New(rand.NewPCG(1337, 420)),
	}
	samples := 0
	for x := range buf.Sample(scoreFunc) {
		t.Log(x)
		// _ = x
		if samples++; 1000 < samples {
			break
		}
	}
}
