package disco

import (
	"testing"
)

func BenchmarkDot(b *testing.B) {
	a := make([]float32, 20)
	c := make([]float32, 20)

	for i := range a {
		a[i] = float32(i)
		c[i] = float32(i)
	}

	for b.Loop() {
		dot(a, c)
	}
}
