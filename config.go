package disco

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

func Factors(factors int) Option {
	return func(c *config) {
		c.factors = factors
	}
}

func Iterations(iterations int) Option {
	return func(c *config) {
		c.iterations = iterations
	}
}

func Regularization(regularization float32) Option {
	return func(c *config) {
		c.regularization = &regularization
	}
}

func LearningRate(learningRate float32) Option {
	return func(c *config) {
		c.learningRate = learningRate
	}
}

func Alpha(alpha float32) Option {
	return func(c *config) {
		c.alpha = alpha
	}
}

func Callback(callback func(info FitInfo)) Option {
	return func(c *config) {
		c.callback = callback
	}
}

func Seed(seed int64) Option {
	return func(c *config) {
		c.seed = seed
	}
}
