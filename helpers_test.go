package disco_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/ankane/disco-go"
)

func assertEqual[T comparable](t *testing.T, exp T, act T) {
	if act != exp {
		t.Errorf("Failed")
	}
}

func assertDeepEqual[T any](t *testing.T, exp T, act T) {
	if !reflect.DeepEqual(act, exp) {
		t.Errorf("Failed")
	}
}

func assertInDelta(t *testing.T, exp float32, act float32, delta float64) {
	diff := math.Abs(float64(exp - act))
	if diff > delta {
		t.Errorf("Failed")
	}
}

func assertNil[T any](t *testing.T, act T) {
	if !reflect.DeepEqual(act, nil) {
		t.Errorf("Failed")
	}
}

func assertContains[T comparable](t *testing.T, haystack []T, needle T) {
	if !contains(haystack, needle) {
		t.Errorf("Failed")
	}
}

func assertNotContains[T comparable](t *testing.T, haystack []T, needle T) {
	if contains(haystack, needle) {
		t.Errorf("Failed")
	}
}

func contains[T comparable](haystack []T, needle T) bool {
	for _, v := range haystack {
		if v == needle {
			return true
		}
	}
	return false
}

func assertError(t *testing.T, err error, message string) {
	if err == nil || err.Error() != message {
		t.Errorf("Failed")
	}
}

func getIds[T disco.Id](recs []disco.Rec[T]) []T {
	ids := make([]T, 0)
	for _, v := range recs {
		ids = append(ids, v.Id)
	}
	return ids
}
