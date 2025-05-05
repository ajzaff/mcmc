package mcmc

import (
	"math/rand/v2"
)

// Buffer is a place to store samples and proposals.
//
// The buffer uses a single slice to store one sample
// followed by a proposal. Therefore the slice length
// should be at least 2*SampleDims.
type Buffer struct {
	SampleDims int
	Buf        []float64
	Theta      []float64
	Rand       *rand.Rand
}
