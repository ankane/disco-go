package disco

import (
	"errors"
	"math"
	"math/rand"
	"sort"
)

type Id interface {
	string | int | uint | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
}

// A recommender.
type Recommender[T Id, U Id] struct {
	userMap     map[T]int
	itemMap     map[U]int
	userIds     []T
	itemIds     []U
	rated       map[int]map[int]bool
	globalMean  float32
	userFactors *matrix
	itemFactors *matrix
	userNorms   []float32
	itemNorms   []float32
}

// A recommendation.
type Rec[T Id] struct {
	Id    T
	Score float32
}

// Information about a training iteration.
type FitInfo struct {
	// The iteration.
	Iteration int
	// The training loss.
	TrainLoss float32
	// The validation loss.
	ValidLoss float32
}

type sparseRow struct {
	index      int
	confidence float32
}

// Creates a recommender with explicit feedback.
func FitExplicit[T Id, U Id](trainSet *Dataset[T, U], options ...Option) (*Recommender[T, U], error) {
	return fit(trainSet, nil, false, options...)
}

// Creates a recommender with implicit feedback.
func FitImplicit[T Id, U Id](trainSet *Dataset[T, U], options ...Option) (*Recommender[T, U], error) {
	return fit(trainSet, nil, true, options...)
}

// Creates a recommender with explicit feedback and performs cross-validation.
func FitEvalExplicit[T Id, U Id](trainSet *Dataset[T, U], validSet *Dataset[T, U], options ...Option) (*Recommender[T, U], error) {
	return fit(trainSet, validSet, false, options...)
}

func fit[T Id, U Id](trainSet *Dataset[T, U], validSet *Dataset[T, U], implicit bool, options ...Option) (*Recommender[T, U], error) {
	config := &config{
		factors:      8,
		iterations:   20,
		learningRate: 0.1,
		alpha:        40.0,
		seed:         rand.Int63(),
	}
	for _, opt := range options {
		opt(config)
	}
	factors := config.factors

	userMap := make(map[T]int, 0)
	itemMap := make(map[U]int, 0)
	userIds := make([]T, 0)
	itemIds := make([]U, 0)
	rated := make(map[int]map[int]bool, 0)

	rowInds := []int{}
	colInds := []int{}
	values := []float32{}

	cui := [][]sparseRow{}
	ciu := [][]sparseRow{}

	if trainSet.Len() == 0 {
		return nil, errors.New("No training data")
	}

	for _, rating := range trainSet.data {
		u, ok := userMap[rating.userId]
		if !ok {
			u = len(userMap)
			userMap[rating.userId] = u
			userIds = append(userIds, rating.userId)
			rated[u] = make(map[int]bool, 0)
		}

		i, ok := itemMap[rating.itemId]
		if !ok {
			i = len(itemMap)
			itemMap[rating.itemId] = i
			itemIds = append(itemIds, rating.itemId)
		}

		if implicit {
			if u == len(cui) {
				cui = append(cui, []sparseRow{})
			}

			if i == len(ciu) {
				ciu = append(ciu, []sparseRow{})
			}

			confidence := 1.0 + config.alpha*rating.value
			cui[u] = append(cui[u], sparseRow{index: i, confidence: confidence})
			ciu[i] = append(ciu[i], sparseRow{index: u, confidence: confidence})
		} else {
			rowInds = append(rowInds, u)
			colInds = append(colInds, i)
			values = append(values, rating.value)
		}

		rated[u][i] = true
	}

	users := len(userMap)
	items := len(itemMap)

	var globalMean float32
	if implicit {
		globalMean = 0.0
	} else {
		var sum float32 = 0.0
		for i := 0; i < len(values); i++ {
			sum += values[i]
		}
		globalMean = sum / float32(len(values))
	}

	var endRange float32
	if implicit {
		endRange = 0.01
	} else {
		endRange = 0.1
	}

	rng := rand.New(rand.NewSource(config.seed))

	userFactors := createFactors(users, factors, rng, endRange)
	itemFactors := createFactors(items, factors, rng, endRange)

	recommender := &Recommender[T, U]{
		userMap:     userMap,
		itemMap:     itemMap,
		userIds:     userIds,
		itemIds:     itemIds,
		rated:       rated,
		globalMean:  globalMean,
		userFactors: userFactors,
		itemFactors: itemFactors,
	}

	if implicit {
		// conjugate gradient method
		// https://www.benfrederickson.com/fast-implicit-matrix-factorization/

		var regularization float32
		if config.regularization != nil {
			regularization = *config.regularization
		} else {
			regularization = 0.01
		}

		for iteration := 0; iteration < config.iterations; iteration++ {
			leastSquaresCg(cui, recommender.userFactors, recommender.itemFactors, regularization)
			leastSquaresCg(ciu, recommender.itemFactors, recommender.userFactors, regularization)

			if config.callback != nil {
				info := FitInfo{
					Iteration: iteration + 1,
					TrainLoss: float32(math.NaN()),
					ValidLoss: float32(math.NaN()),
				}
				config.callback(info)
			}
		}
	} else {
		// stochastic gradient method with twin learners
		// https://www.csie.ntu.edu.tw/~cjlin/papers/libmf/mf_adaptive_pakdd.pdf
		// algorithm 2

		learningRate := config.learningRate
		var lambda float32
		if config.regularization != nil {
			lambda = *config.regularization
		} else {
			lambda = 0.1
		}
		k := factors
		ks := int(math.Max(math.Round(float64(k)*0.08), 1))

		gSlow := make([]float32, users)
		gFast := make([]float32, users)
		for i := 0; i < users; i++ {
			gSlow[i] = 1.0
			gFast[i] = 1.0
		}

		hSlow := make([]float32, items)
		hFast := make([]float32, items)
		for i := 0; i < items; i++ {
			hSlow[i] = 1.0
			hFast[i] = 1.0
		}

		for iteration := 0; iteration < config.iterations; iteration++ {
			var trainLoss float32 = 0.0

			for _, j := range rand.Perm(trainSet.Len()) {
				u := rowInds[j]
				v := colInds[j]

				pu := userFactors.Row(u)
				qv := itemFactors.Row(v)
				e := values[j] - dot(pu, qv)

				// slow learner
				var gHat float32 = 0.0
				var hHat float32 = 0.0

				nu := learningRate / sqrt(gSlow[u])
				nv := learningRate / sqrt(hSlow[v])

				for d := 0; d < ks; d++ {
					gud := -e*qv[d] + lambda*pu[d]
					hvd := -e*pu[d] + lambda*qv[d]

					gHat += gud * gud
					hHat += hvd * hvd

					pu[d] -= nu * gud
					qv[d] -= nv * hvd
				}

				gSlow[u] += gHat / float32(ks)
				hSlow[v] += hHat / float32(ks)

				// fast learner
				// don't update on first outer iteration
				if iteration > 0 {
					var gHat float32 = 0.0
					var hHat float32 = 0.0

					nu := learningRate / sqrt(gFast[u])
					nv := learningRate / sqrt(hFast[v])

					for d := ks; d < k; d++ {
						gud := -e*qv[d] + lambda*pu[d]
						hvd := -e*pu[d] + lambda*qv[d]

						gHat += gud * gud
						hHat += hvd * hvd

						pu[d] -= nu * gud
						qv[d] -= nv * hvd
					}

					gFast[u] += gHat / float32(k-ks)
					hFast[v] += hHat / float32(k-ks)
				}

				trainLoss += e * e
			}

			if config.callback != nil {
				trainLoss = sqrt(trainLoss / float32(trainSet.Len()))

				var validLoss float32
				if validSet != nil {
					validLoss = recommender.Rmse(validSet)
				} else {
					validLoss = 0.0
				}

				info := FitInfo{
					Iteration: iteration + 1,
					TrainLoss: trainLoss,
					ValidLoss: validLoss,
				}
				config.callback(info)
			}
		}
	}

	return recommender, nil
}

// Returns recommendations for a user.
func (r *Recommender[T, U]) UserRecs(userId T, count int) []Rec[U] {
	u, ok := r.userMap[userId]
	if !ok {
		return []Rec[U]{}
	}

	rated := r.rated[u]
	factors := r.userFactors.Row(u)
	predictions := make([]Rec[int], 0, r.itemFactors.rows)
	for j := 0; j < r.itemFactors.rows; j++ {
		predictions = append(predictions, Rec[int]{Id: j, Score: dot(factors, r.itemFactors.Row(j))})
	}
	sort.Slice(predictions, func(j, k int) bool {
		return predictions[j].Score > predictions[k].Score
	})
	predictions = first(predictions, count+len(rated))

	recs := make([]Rec[U], 0, count+len(rated))
	for _, prediction := range predictions {
		_, ok := rated[prediction.Id]
		if !ok {
			recs = append(recs, Rec[U]{Id: r.itemIds[prediction.Id], Score: prediction.Score})
		}
	}
	return first(recs, count)
}

// Returns recommendations for an item.
func (r *Recommender[T, U]) ItemRecs(itemId U, count int) []Rec[U] {
	if r.itemNorms == nil {
		r.itemNorms = r.itemFactors.Norms()
	}
	return similar(r.itemMap, r.itemIds, r.itemFactors, r.itemNorms, itemId, count)
}

// Returns similar users.
func (r *Recommender[T, U]) SimilarUsers(userId T, count int) []Rec[T] {
	if r.userNorms == nil {
		r.userNorms = r.userFactors.Norms()
	}
	return similar(r.userMap, r.userIds, r.userFactors, r.userNorms, userId, count)
}

// Returns the predicted rating for a specific user and item.
func (r *Recommender[T, U]) Predict(userId T, itemId U) float32 {
	u, ok := r.userMap[userId]
	if !ok {
		return r.globalMean
	}

	i, ok := r.itemMap[itemId]
	if !ok {
		return r.globalMean
	}

	return dot(r.userFactors.Row(u), r.itemFactors.Row(i))
}

// Returns user ids.
func (r *Recommender[T, U]) UserIds() []T {
	return r.userIds
}

// Returns item ids.
func (r *Recommender[T, U]) ItemIds() []U {
	return r.itemIds
}

// Returns factors for a specific user.
func (r *Recommender[T, U]) UserFactors(userId T) []float32 {
	u, ok := r.userMap[userId]
	if !ok {
		return nil
	}
	return r.userFactors.Row(u)
}

// Returns factors for a specific item.
func (r *Recommender[T, U]) ItemFactors(itemId U) []float32 {
	i, ok := r.itemMap[itemId]
	if !ok {
		return nil
	}
	return r.itemFactors.Row(i)
}

// Returns the global mean.
func (r *Recommender[T, U]) GlobalMean() float32 {
	return r.globalMean
}

// Calculates the root mean square error for a dataset.
func (r *Recommender[T, U]) Rmse(data *Dataset[T, U]) float32 {
	var sum float32 = 0.0
	for _, v := range data.data {
		diff := r.Predict(v.userId, v.itemId) - v.value
		sum += diff * diff
	}
	return sqrt(sum / float32(len(data.data)))
}

func leastSquaresCg(cui [][]sparseRow, x *matrix, y *matrix, regularization float32) {
	cgSteps := 3

	// calculate YtY
	factors := y.cols
	yty := newMatrix(factors, factors)
	for i := 0; i < factors; i++ {
		for j := 0; j < factors; j++ {
			var sum float32 = 0.0
			for k := 0; k < y.rows; k++ {
				sum += y.data[k*factors+i] * y.data[k*factors+j]
			}
			yty.data[i*factors+j] = sum
		}
	}
	for i := 0; i < factors; i++ {
		yty.data[i*factors+i] += regularization
	}

	for u, rowVec := range cui {
		// start from previous iteration
		xi := x.Row(u)

		// calculate residual r = (YtCuPu - (YtCuY.dot(Xu), without computing YtCuY
		r := yty.Dot(xi)
		neg(r)
		for _, row := range rowVec {
			i := row.index
			var confidence float32 = row.confidence
			scaledAdd(r, confidence-(confidence-1.0)*dot(y.Row(i), xi), y.Row(i))
		}

		p := make([]float32, factors)
		copy(p, r)
		rsold := dot(r, r)

		for j := 0; j < cgSteps; j++ {
			// calculate Ap = YtCuYp - without actually calculating YtCuY
			ap := yty.Dot(p)
			for _, row := range rowVec {
				i := row.index
				var confidence float32 = row.confidence
				scaledAdd(ap, (confidence-1.0)*dot(y.Row(i), p), y.Row(i))
			}

			// standard CG update
			alpha := rsold / dot(p, ap)
			scaledAdd(xi, alpha, p)
			scaledAdd(r, -alpha, ap)
			rsnew := dot(r, r)

			if rsnew < 1e-20 {
				break
			}

			rs := rsnew / rsold
			for i := 0; i < len(p); i++ {
				p[i] = r[i] + rs*p[i]
			}
			rsold = rsnew
		}
	}
}

func createFactors(rows int, cols int, rng *rand.Rand, endRange float32) *matrix {
	m := newMatrix(rows, cols)
	for i := 0; i < rows*cols; i++ {
		m.data[i] = rng.Float32() * endRange
	}
	return m
}

func similar[T Id](idMap map[T]int, ids []T, factors *matrix, norms []float32, id T, count int) []Rec[T] {
	i, ok := idMap[id]
	if !ok {
		return []Rec[T]{}
	}

	rowFactors := factors.Row(i)
	rowNorm := norms[i]

	predictions := make([]Rec[int], 0, factors.rows)
	for j := 0; j < factors.rows; j++ {
		denom := rowNorm * norms[j]
		if denom == 0 {
			denom = 0.00001
		}
		predictions = append(predictions, Rec[int]{Id: j, Score: dot(rowFactors, factors.Row(j)) / denom})
	}
	sort.Slice(predictions, func(j, k int) bool {
		return predictions[j].Score > predictions[k].Score
	})
	predictions = first(predictions, count+1)

	recs := make([]Rec[T], 0, count+1)
	for _, prediction := range predictions {
		if prediction.Id != i {
			recs = append(recs, Rec[T]{Id: ids[prediction.Id], Score: prediction.Score})
		}
	}
	return first(recs, count)
}

func dot(a []float32, b []float32) float32 {
	var d float32 = 0.0
	for i := 0; i < len(a); i++ {
		d += a[i] * b[i]
	}
	return d
}

func scaledAdd(x []float32, a float32, v []float32) {
	for i := 0; i < len(x); i++ {
		x[i] += a * v[i]
	}
}

func neg(x []float32) {
	for i := 0; i < len(x); i++ {
		x[i] = -x[i]
	}
}

func sqrt(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

func first[T any](s []T, n int) []T {
	if n > len(s) {
		return s
	}
	return s[:n]
}
