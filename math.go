//go:build !goexperiment.simd

package disco

func dot(a []float32, b []float32) float32 {
	var distance float32 = 0.0
	for i, ai := range a {
		distance += ai * b[i]
	}
	return distance
}
