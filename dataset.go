package disco

import (
	"math/rand"
	"slices"
)

type Dataset[T Id, U Id] struct {
	data []rating[T, U]
}

type rating[T Id, U Id] struct {
	userId T
	itemId U
	value  float32
}

func NewDataset[T Id, U Id]() *Dataset[T, U] {
	return &Dataset[T, U]{data: []rating[T, U]{}}
}

func (d *Dataset[T, U]) Push(userId T, itemId U, value float32) {
	d.data = append(d.data, rating[T, U]{userId: userId, itemId: itemId, value: value})
}

func (d *Dataset[T, U]) Len() int {
	return len(d.data)
}

func (d *Dataset[T, U]) Grow(n int) {
	d.data = slices.Grow(d.data, n)
}

func (d *Dataset[T, U]) SplitRandom(p float32) (*Dataset[T, U], *Dataset[T, U]) {
	index := int(p * float32(len(d.data)))
	data := make([]rating[T, U], len(d.data))
	for i, v := range rand.Perm(len(d.data)) {
		data[v] = d.data[i]
	}
	trainSet := &Dataset[T, U]{data: data[:index]}
	validSet := &Dataset[T, U]{data: data[index:]}
	return trainSet, validSet
}
