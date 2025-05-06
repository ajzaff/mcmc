package mcmc

import (
	"math/rand/v2"
)

// Buffer is a place to store samples and proposals.
//
// The buffer uses a single slice to store one sample
// followed by a proposal. Therefore the slice length
// should be 2*SampleDims.
type Buffer struct {
	SampleDims int
	Buf        []float64
	Theta      []float64
	LastScore  float64
	Rand       *rand.Rand
}

// LastSample returns the last sample from the Buffer.
func (b *Buffer) LastSample() []float64 { return b.Buf[0:b.SampleDims] }
