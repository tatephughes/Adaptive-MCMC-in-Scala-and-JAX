import breeze.plot._
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
//import breeze.stats.distributions.Rand.FixedSeed.randBasis
import org.apache.commons.math3.random.MersenneTwister

implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(42L)))


case class AM_state(j: Double, 
                    x_sum: DenseVector[Double], 
                    xxt_sum: DenseMatrix[Double],
                    x: DenseVector[Double])



object AdaptiveMetropolis:

  // updated from Darren's scala course, github.com/darrenjw/scala-course
  def thin[T](chain:LazyList[T], th: Int): LazyList[T] = {
    val droppedchain = chain.drop(th - 1)
    if (droppedchain.isEmpty) LazyList.empty else
      droppedchain.head #:: thin(droppedchain.tail, th)
  }

  def plotter(sample: Array[DenseVector[Double]], // the sample to plot
              j: Int, // the coordinate to plot
              file_path: String): Unit = {

    val y = DenseVector(sample.map(x => x(j)))
    val x = DenseVector.tabulate(y.length)(i => (i+1).toDouble)

    val f = Figure()
    val p = f.subplot(0)

    p += plot(x,y)
    p.xlabel = "Index"
    p.ylabel = "x_j"

    p.title = "Trace Plot of x_j"

    f.saveas(file_path)

  }
  
  def one_AMRTH_step(state: AM_state, q: DenseMatrix[Double], r: DenseMatrix[Double], prog: Boolean): AM_state = {

    val j = state.j
    val x_sum = state.x_sum
    val xxt_sum = state.xxt_sum
    val x = state._4

    // print progress every 1000 iterations if 'prog=true'
    if (j % 1000 == 0 && prog) {
      print("\n   Running: " + j + "th iteration\n")
      //print("   Completed " + j/10000 + "%\n")
      val runtime = Runtime.getRuntime()
      print(s"** Used Memory (MB): ${(runtime.totalMemory-runtime.freeMemory)/(1048576)}")
    }

    val d = x.length

    if (j <= 2*d) then { // procedure for n<=2d

      val proposed_move = x.map((xi:Double) => Gaussian(xi, 1/d.toDouble).sample())
      val alpha = 0.5 * ((x.t * (r \ (q.t * x))) - (proposed_move.t * (r \ (q.t * proposed_move))))
      val log_acceptance_prob = math.min(0.0, alpha)
      val u = Uniform(0,1).draw()

      if (math.log(u) < log_acceptance_prob) then {
        val nx_sum = x_sum + proposed_move
        val nxxt_sum = xxt_sum + (proposed_move * proposed_move.t)
        return(AM_state(j+1, nx_sum, nxxt_sum, proposed_move))
      } else {
        val nx_sum = x_sum + x
        val nxxt_sum = xxt_sum + (x * x.t)
        return(AM_state(j+1, nx_sum, nxxt_sum, x))
      }

    } else { // the actually adaptive part

      val sigma_j = (xxt_sum / j)
                    - ((x_sum * x_sum.t) / (j*j))

      val u1 = Uniform(0,1).draw()

      val proposed_move = if (u1 < 0.95) then {
        MultivariateGaussian(x, sigma_j * (2.38*2.38/d.toDouble)).draw()
      } else {
        x.map((xi:Double) => Gaussian(xi, 0.01/d.toDouble).sample())
      }

      val alpha = 0.5 * ((x.t * (r \ (q.t * x))) - (proposed_move.t * (r \ (q.t * proposed_move))))

      val log_acceptance_prob = math.min(0.0, alpha)
      val u2 = Uniform(0,1).draw()

      if (math.log(u2) < log_acceptance_prob) then {
        val nx_sum = x_sum + proposed_move
        val nxxt_sum = xxt_sum + (proposed_move * proposed_move.t)
        return(AM_state(j+1, nx_sum, nxxt_sum, proposed_move))
      } else {
        val nx_sum = x_sum + x
        val nxxt_sum = xxt_sum + (x * x.t)
        return(AM_state(j+1, nx_sum, nxxt_sum, x))
      }
    }

  }

  def AMRTH(state0: AM_state, sigma: DenseMatrix[Double], prog: Boolean): LazyList[AM_state] = {

    val qr.QR(q,r) = qr(sigma)

    LazyList.iterate(state0)((state: AM_state) => one_AMRTH_step(state, q, r, prog))
  }

  @main def run(): Unit =

    // dimension of the state space
    val d = 25

    // create a chaotic variance to target
    val data = Gaussian(0,1).sample(d*d).toArray.grouped(d).toArray
    val M = DenseMatrix(data: _*)
    val sigma = M.t * M

    // initial state
    val state0 = AM_state(0.0, DenseVector.zeros[Double](d), DenseMatrix.eye[Double](d), DenseVector.zeros[Double](d))

    val n: Int = 10000 // size of the desired sample
    val burnin: Int = 10000
    val thinrate: Int = 10
    // The actual number of iterations computed is n/thin + burnin

    val amrth_sample = thin(AMRTH(state0, sigma, true).map(_.x).drop(burnin),thinrate).take(n).toArray

    // Empirical Variance matrix of the sample
    val sigma_j = cov(DenseMatrix(amrth_sample: _*))

    val eigsigmaj = eig(sigma_j).eigenvalues
    val eigsigma  = eig(sigma).eigenvalues

    val lambda = sqrt(eigsigmaj) *:* sqrt(eigsigma).map(x => 1/x)

    val lambdaminus2sum = sum(lambda.map(x => 1/(x*x)))
    val lambdainvsum = sum(lambda.map(x => 1/x))

    // According to Roberts and Rosenthal, this should go to 1 at the stationary distribution
    val b = d * (lambdaminus2sum / (lambdainvsum*lambdainvsum))

    print("\nThe true variance of x_1 value is\n" + sigma(1,1))

    print("\n\nThe Empirical sigma value is\n" + sigma_j(1,1))

    print("\n The b value is " + b)

    plotter(amrth_sample, 0, "./exports/adaptive_trace.png")
