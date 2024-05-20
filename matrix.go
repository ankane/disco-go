package disco

type matrix struct {
	rows int
	cols int
	data []float32
}

func newMatrix(rows int, cols int) *matrix {
	data := make([]float32, rows*cols)
	return &matrix{rows: rows, cols: cols, data: data}
}

func (m *matrix) Row(row int) []float32 {
	start := row * m.cols
	return m.data[start : start+m.cols]
}

func (m *matrix) Dot(x []float32) []float32 {
	res := make([]float32, m.rows)
	for i := 0; i < m.rows; i++ {
		var sum float32 = 0.0
		row := m.Row(i)
		for j := 0; j < m.cols; j++ {
			sum += row[j] * x[j]
		}
		res[i] = sum
	}
	return res
}

func (m *matrix) Norms() []float32 {
	res := make([]float32, 0, m.rows)

	for i := 0; i < m.rows; i++ {
		row := m.Row(i)
		var norm float32 = 0.0
		for j := 0; j < len(row); j++ {
			norm += row[j] * row[j]
		}
		norm = sqrt(norm)
		res = append(res, norm)
	}

	return res
}
