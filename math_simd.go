//go:build goexperiment.simd

package disco

import (
	"simd/archsimd"
)

func dot(a []float32, b []float32) float32 {
	var dist archsimd.Float32x4

	if len(b) != len(a) {
		panic("")
	}

	for len(a) >= 4 {
		axs := archsimd.LoadFloat32x4(a[:4])
		bxs := archsimd.LoadFloat32x4(b[:4])
		dist = axs.MulAdd(bxs, dist)
		a = a[4:]
		b = b[4:]
	}

	var s [4]float32
	dist.StoreArray(&s)
	distance := s[0] + s[1] + s[2] + s[3]

	for i, ai := range a {
		distance += ai * b[i]
	}

	return distance
}
