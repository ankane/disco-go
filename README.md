# Disco Go

:fire: Recommendations for Go using collaborative filtering

- Supports user-based and item-based recommendations
- Works with explicit and implicit feedback
- Uses high-performance matrix factorization

[![Build Status](https://github.com/ankane/disco-go/actions/workflows/build.yml/badge.svg)](https://github.com/ankane/disco-go/actions)

## Installation

Run:

```sh
go get github.com/ankane/disco-go
```

## Getting Started

Import the package

```go
import "github.com/ankane/disco-go"
```

Prep your data in the format `userId, itemId, value`

```go
data := disco.NewDataset[string, string]()
data.Push("user_a", "item_a", 5.0)
data.Push("user_a", "item_b", 3.5)
data.Push("user_b", "item_a", 4.0)
```

IDs can be integers or strings

```go
data.Push(1, "item_a", 5.0)
```

If users rate items directly, this is known as explicit feedback. Fit the recommender with:

```go
recommender, err := disco.FitExplicit(data)
```

If users don’t rate items directly (for instance, they’re purchasing items or reading posts), this is known as implicit feedback. Use `1.0` or a value like number of purchases or page views for the dataset, and fit the recommender with:

```go
recommender, err := disco.FitImplicit(data)
```

Get user-based recommendations - “users like you also liked”

```go
recommender.UserRecs(userId, 5)
```

Get item-based recommendations - “users who liked this item also liked”

```go
recommender.ItemRecs(itemId, 5)
```

Get predicted ratings for a specific user and item

```go
recommender.Predict(userId, itemId)
```

Get similar users

```go
recommender.SimilarUsers(userId, 5)
```

## Examples

### MovieLens

Load the data

```go
data, err := disco.LoadMovieLens()
```

Create a recommender

```go
recommender, err := disco.FitExplicit(data, disco.Factors(20))
```

Get similar movies

```go
recommender.ItemRecs("Star Wars (1977)", 5)
```

## Storing Recommendations

Save recommendations to your database.

Alternatively, you can store only the factors and use a library like [pgvector-go](https://github.com/pgvector/pgvector-go). See an [example](https://github.com/pgvector/pgvector-go/blob/master/examples/disco_test.go).

## Algorithms

Disco uses high-performance matrix factorization.

- For explicit feedback, it uses the [stochastic gradient method with twin learners](https://www.csie.ntu.edu.tw/~cjlin/papers/libmf/mf_adaptive_pakdd.pdf)
- For implicit feedback, it uses the [conjugate gradient method](https://www.benfrederickson.com/fast-implicit-matrix-factorization/)

Specify the number of factors and iterations

```go
recommender, err := disco.FitExplicit(data, disco.Factors(8), disco.Iterations(20))
```

## Progress

Pass a callback to show progress

```go
callback := func(info disco.FitInfo) { fmt.Printf("%+v\n", info) }
recommender, err := disco.FitExplicit(data, disco.Callback(callback))
```

Note: `TrainLoss` and `ValidLoss` are not available for implicit feedback

## Validation

Pass a validation set with explicit feedback

```go
recommender, err := disco.FitEvalExplicit(trainSet, validSet)
```

The loss function is RMSE

## Cold Start

Collaborative filtering suffers from the [cold start problem](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)). It’s unable to make good recommendations without data on a user or item, which is problematic for new users and items.

```go
recommender.UserRecs(newUserId, 5) // returns empty array
```

There are a number of ways to deal with this, but here are some common ones:

- For user-based recommendations, show new users the most popular items
- For item-based recommendations, make content-based recommendations

## Reference

Get ids

```go
recommender.UserIds()
recommender.ItemIds()
```

Get the global mean

```go
recommender.GlobalMean()
```

Get factors

```go
recommender.UserFactors(userId)
recommender.ItemFactors(itemId)
```

## References

- [A Learning-rate Schedule for Stochastic Gradient Methods to Matrix Factorization](https://www.csie.ntu.edu.tw/~cjlin/papers/libmf/mf_adaptive_pakdd.pdf)
- [Faster Implicit Matrix Factorization](https://www.benfrederickson.com/fast-implicit-matrix-factorization/)

## History

View the [changelog](https://github.com/ankane/disco-go/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/disco-go/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/disco-go/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/disco-go.git
cd disco-go
go mod tidy
go test -v
```
