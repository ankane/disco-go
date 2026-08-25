//go:build goexperiment.simd

package disco

import (
	"simd/archsimd"
)

func dot(a []float32, b []float32) float32 {
	dim := len(a)
	if len(b) < dim {
		dim = len(b)
	}
	i := 0
	count := dim / 8 * 8
	var dist archsimd.Float32x8

	for ; i < count; i += 8 {
		axs := archsimd.LoadFloat32x8(a[i : i+8])
		bxs := archsimd.LoadFloat32x8(b[i : i+8])
		dist = axs.MulAdd(bxs, dist)
	}

	var s [8]float32
	dist.StoreArray(&s)
	distance := s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7]

	for i < dim {
		distance += a[i] * b[i]
	}

	return distance
}
