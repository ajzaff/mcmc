package mcmc

import (
	"math/rand/v2"
	"testing"
)

func BenchmarkBufferSample(b *testing.B) {
	buf := Buffer{
		SampleDims: 1,
		Buf:        make([]float64, 2),
		Theta:      []float64{1},
		Rand:       rand.New(rand.NewPCG(1337, 420)),
	}
	b.ResetTimer()
	for b.Loop() {
		buf.Sample(scoreFunc)
	}
}
