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
	count := dim / 4 * 4
	var dist archsimd.Float32x4

	for ; i < count; i += 4 {
		axs := archsimd.LoadFloat32x4(a[i : i+4])
		bxs := archsimd.LoadFloat32x4(b[i : i+4])
		dist = axs.MulAdd(bxs, dist)
	}

	var s [4]float32
	dist.StoreArray(&s)
	distance := s[0] + s[1] + s[2] + s[3]

	for i < dim {
		distance += a[i] * b[i]
	}

	return distance
}
