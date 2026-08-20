package disco

import (
	"math/rand/v2"
	"slices"
)

type denseMatrix struct {
	rows int
	cols int
	data []float32
}

func newDenseMatrix(rows int, cols int) *denseMatrix {
	data := make([]float32, rows*cols)
	return &denseMatrix{rows: rows, cols: cols, data: data}
}

func (m *denseMatrix) Row(row int) []float32 {
	start := row * m.cols
	return m.data[start : start+m.cols]
}

func (m *denseMatrix) Dot(x []float32) []float32 {
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

func (m *denseMatrix) Norms() []float32 {
	res := make([]float32, 0, m.rows)

	for i := 0; i < m.rows; i++ {
		row := m.Row(i)
		var norm float32 = 0.0
		for j := range row {
			norm += row[j] * row[j]
		}
		norm = sqrt(norm)
		res = append(res, norm)
	}

	return res
}

type cooMatrix struct {
	rowInds []int
	colInds []int
	values  []float32
}

func newCooMatrix() *cooMatrix {
	return &cooMatrix{rowInds: []int{}, colInds: []int{}, values: []float32{}}
}

func (m *cooMatrix) Grow(len int) {
	m.rowInds = slices.Grow(m.rowInds, len)
	m.colInds = slices.Grow(m.colInds, len)
	m.values = slices.Grow(m.values, len)
}

func (m *cooMatrix) Push(u int, i int, value float32) {
	m.rowInds = append(m.rowInds, u)
	m.colInds = append(m.colInds, i)
	m.values = append(m.values, value)
}

func (m *cooMatrix) Len() int {
	return len(m.rowInds)
}

func (m *cooMatrix) Get(i int) (int, int, float32) {
	return m.rowInds[i], m.colInds[i], m.values[i]
}

func (m *cooMatrix) Shuffle(rng *rand.Rand) {
	rng.Shuffle(m.Len(), func(i, j int) {
		m.rowInds[i], m.rowInds[j] = m.rowInds[j], m.rowInds[i]
		m.colInds[i], m.colInds[j] = m.colInds[j], m.colInds[i]
		m.values[i], m.values[j] = m.values[j], m.values[i]
	})
}

type lilElement struct {
	index int
	value float32
}

type lilMatrix struct {
	rowList [][]lilElement
}

func newLilMatrix() *lilMatrix {
	return &lilMatrix{rowList: [][]lilElement{}}
}

func (m *lilMatrix) Push(rowInd int, colInd int, value float32) {
	if rowInd == len(m.rowList) {
		m.rowList = append(m.rowList, []lilElement{})
	}
	m.rowList[rowInd] = append(m.rowList[rowInd], lilElement{index: colInd, value: value})
}
