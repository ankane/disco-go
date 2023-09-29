package disco_test

import (
	"sort"
	"testing"

	"github.com/ankane/disco-go"
)

func TestExplicit(t *testing.T) {
	data, err := disco.LoadMovieLens()
	assertNil(t, err)

	recommender, err := disco.FitExplicit(data, disco.Factors(20))
	assertNil(t, err)

	recs := recommender.ItemRecs("Star Wars (1977)", 5)
	assertEqual(t, 5, len(recs))

	itemIds := getIds(recs)
	assertContains(t, itemIds, "Empire Strikes Back, The (1980)")
	assertContains(t, itemIds, "Return of the Jedi (1983)")
	assertNotContains(t, itemIds, "Star Wars (1977)")

	assertInDelta(t, 0.9972, recs[0].Score, 0.01)
}

func TestImplicit(t *testing.T) {
	data, err := disco.LoadMovieLens()
	assertNil(t, err)

	recommender, err := disco.FitImplicit(data, disco.Factors(20))
	assertNil(t, err)

	assertEqual(t, 0.0, recommender.GlobalMean())

	recs := recommender.ItemRecs("Star Wars (1977)", 20)
	itemIds := getIds(recs)
	assertContains(t, itemIds, "Empire Strikes Back, The (1980)")
	assertContains(t, itemIds, "Return of the Jedi (1983)")
	assertNotContains(t, itemIds, "Star Wars (1977)")
}

func TestRated(t *testing.T) {
	data := disco.NewDataset[int, string]()
	data.Push(1, "A", 1.0)
	data.Push(1, "B", 1.0)
	data.Push(1, "C", 1.0)
	data.Push(1, "D", 1.0)
	data.Push(2, "C", 1.0)
	data.Push(2, "D", 1.0)
	data.Push(2, "E", 1.0)
	data.Push(2, "F", 1.0)

	recommender, err := disco.FitExplicit(data)
	assertNil(t, err)

	itemIds := getIds(recommender.UserRecs(1, 5))
	sort.Strings(itemIds)
	assertDeepEqual(t, []string{"E", "F"}, itemIds)

	itemIds = getIds(recommender.UserRecs(2, 5))
	sort.Strings(itemIds)
	assertDeepEqual(t, []string{"A", "B"}, itemIds)
}

func TestItemRecsSameScore(t *testing.T) {
	data := disco.NewDataset[int, string]()
	data.Push(1, "A", 1.0)
	data.Push(1, "B", 1.0)
	data.Push(2, "C", 1.0)

	recommender, err := disco.FitExplicit(data)
	assertNil(t, err)

	itemIds := getIds(recommender.ItemRecs("A", 5))
	assertDeepEqual(t, []string{"B", "C"}, itemIds)
}

func TestSimilarUsers(t *testing.T) {
	data, err := disco.LoadMovieLens()
	assertNil(t, err)

	recommender, err := disco.FitExplicit(data)
	assertNil(t, err)

	assertEqual(t, 5, len(recommender.SimilarUsers(1, 5)))
	assertEqual(t, 0, len(recommender.SimilarUsers(100000, 5)))
}

func TestIds(t *testing.T) {
	data := disco.NewDataset[int, string]()
	data.Push(1, "A", 1.0)
	data.Push(1, "B", 1.0)
	data.Push(2, "B", 1.0)

	recommender, err := disco.FitExplicit(data)
	assertNil(t, err)

	assertDeepEqual(t, []int{1, 2}, recommender.UserIds())
	assertDeepEqual(t, []string{"A", "B"}, recommender.ItemIds())
}

func TestFactors(t *testing.T) {
	data := disco.NewDataset[int, string]()
	data.Push(1, "A", 1.0)
	data.Push(1, "B", 1.0)
	data.Push(2, "B", 1.0)

	recommender, err := disco.FitExplicit(data, disco.Factors(20))
	assertNil(t, err)

	assertEqual(t, 20, len(recommender.UserFactors(1)))
	assertEqual(t, 20, len(recommender.ItemFactors("A")))

	assertDeepEqual(t, nil, recommender.UserFactors(3))
	assertDeepEqual(t, nil, recommender.ItemFactors("C"))
}

func TestValidationSetExplicit(t *testing.T) {
	data, err := disco.LoadMovieLens()
	assertNil(t, err)

	trainSet, validSet := data.SplitRandom(0.8)
	assertEqual(t, 80000, trainSet.Len())
	assertEqual(t, 20000, validSet.Len())

	var lastInfo disco.FitInfo
	callback := func(info disco.FitInfo) { lastInfo = info }
	_, err = disco.FitEvalExplicit(trainSet, validSet, disco.Callback(callback))
	assertNil(t, err)

	assertInDelta(t, 0.92, lastInfo.ValidLoss, 0.2)
}

func TestUserRecsNewUser(t *testing.T) {
	data := disco.NewDataset[int, int]()
	data.Push(1, 1, 5.0)
	data.Push(2, 1, 3.0)

	recommender, err := disco.FitExplicit(data)
	assertNil(t, err)

	assertEqual(t, 0, len(recommender.UserRecs(1000, 5)))
}

func TestPredict(t *testing.T) {
	data, err := disco.LoadMovieLens()
	assertNil(t, err)

	trainSet, validSet := data.SplitRandom(0.8)
	assertEqual(t, 80000, trainSet.Len())
	assertEqual(t, 20000, validSet.Len())

	recommender, err := disco.FitEvalExplicit(trainSet, validSet, disco.Factors(20))
	assertNil(t, err)

	recommender.Predict(1, "Star Wars (1977)")
}

func TestPredictNewUser(t *testing.T) {
	data, err := disco.LoadMovieLens()
	assertNil(t, err)

	recommender, err := disco.FitExplicit(data, disco.Factors(20))
	assertNil(t, err)

	assertInDelta(t, recommender.GlobalMean(), recommender.Predict(100000, "Star Wars (1977)"), 0.001)
}

func TestPredictNewItem(t *testing.T) {
	data, err := disco.LoadMovieLens()
	assertNil(t, err)

	recommender, err := disco.FitExplicit(data, disco.Factors(20))
	assertNil(t, err)

	assertInDelta(t, recommender.GlobalMean(), recommender.Predict(1, "New movie"), 0.001)
}

func TestCallback(t *testing.T) {
	data := disco.NewDataset[int, int]()
	data.Push(1, 1, 5.0)

	iterations := 0
	callback := func(info disco.FitInfo) { iterations += 1 }
	_, err := disco.FitExplicit(data, disco.Callback(callback))
	assertNil(t, err)

	assertEqual(t, 20, iterations)
}

func TestNoTrainingData(t *testing.T) {
	data := disco.NewDataset[int, string]()
	_, err := disco.FitExplicit(data)
	assertError(t, err, "No training data")
}
