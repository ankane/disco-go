package disco

// A recommender option.
type Option func(*config)

type config struct {
	factors        int
	iterations     int
	regularization *float32
	learningRate   float32
	alpha          float32
	callback       func(info FitInfo)
	seed           int64
}

// Sets the number of factors.
func Factors(factors int) Option {
	return func(c *config) {
		c.factors = factors
	}
}

// Sets the number of iterations.
func Iterations(iterations int) Option {
	return func(c *config) {
		c.iterations = iterations
	}
}

// Sets the regularization.
func Regularization(regularization float32) Option {
	return func(c *config) {
		c.regularization = &regularization
	}
}

// Sets the learning rate.
func LearningRate(learningRate float32) Option {
	return func(c *config) {
		c.learningRate = learningRate
	}
}

// Sets alpha.
func Alpha(alpha float32) Option {
	return func(c *config) {
		c.alpha = alpha
	}
}

// Sets the callback for each iteration.
func Callback(callback func(info FitInfo)) Option {
	return func(c *config) {
		c.callback = callback
	}
}

// Sets the random seed.
func Seed(seed int64) Option {
	return func(c *config) {
		c.seed = seed
	}
}
