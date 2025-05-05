package mcmc

import (
	"iter"
	"math/rand/v2"
	"testing"
)

func BenchmarkBufferSample(b *testing.B) {
	buf := Buffer{
		SampleDims: 1,
		Buf:        make([]float64, 2),
		Theta:      []float64{1},
	}
	samples := buf.Sample(scoreFunc)
	buf.Rand = rand.New(rand.NewPCG(1337, 420))
	next, _ := iter.Pull(samples)
	for b.Loop() {
		next()
	}
}
