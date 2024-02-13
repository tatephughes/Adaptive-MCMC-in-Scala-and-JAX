import breeze.plot._
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
//import breeze.stats.distributions.Rand.FixedSeed.randBasis
import org.apache.commons.math3.random.MersenneTwister

case class AM_state(j: Double, 
                    x_sum: DenseVector[Double], 
                    xxt_sum: DenseMatrix[Double],
                    x: DenseVector[Double])



object AdaptiveMetropolis:

  def plotter(sample: LazyList[DenseVector[Double]], 
              n: Int, 
              j: Int,
              file_path: String): Unit = {

    val xvals = Array.tabulate(n)(i => i.toDouble)
    val yvals = sample.map((x: DenseVector[Double]) => x(0)).take(n).toArray


    val f = Figure()
    val p = f.subplot(0)


    p += plot(xvals,yvals)
    p.xlabel = "Index"
    p.ylabel = "x_1"

    p.title = "Trace Plot of x_j"

    f.saveas(file_path)

  }
  
  def one_AMRTH_step(state: AM_state, q: DenseMatrix[Double], r: DenseMatrix[Double], seed: Long): AM_state = {

    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))

    val j = state.j
    val x_sum = state.x_sum
    val xxt_sum = state.xxt_sum
    val x = state._4

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

  def AMRTH(state0: AM_state, sigma: DenseMatrix[Double], seed: Long): LazyList[AM_state] = {

    val qr.QR(q,r) = qr(sigma)

    LazyList.iterate(state0)((state: AM_state) => one_AMRTH_step(state, q, r, seed + state.j.toLong))
  }

  @main def run(): Unit =

    val seed: Long = 42L

    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))

    val d = 25

    val data = Gaussian(0,1).sample(d*d).toArray.grouped(d).toArray
  
    val M = DenseMatrix(data: _*)
    val sigma = M.t * M

    val state0 = AM_state(0.0, DenseVector.zeros[Double](d), DenseMatrix.eye[Double](d), DenseVector.zeros[Double](d))

    val amrth_sample = AMRTH(state0, sigma, seed)

    val n: Int = 100000

    val xxt_sum = amrth_sample(n).xxt_sum
    val x_sum = amrth_sample(n).x_sum
    
    val sigma_j = (xxt_sum / n.toDouble) - ((x_sum * x_sum.t) / (n*n).toDouble)

    //val eigvalues = eig(sqrtm(sigma_j)*sqrtm(inv(sigma))).eigenvalues

    //val eigminussquare = eigvalues.map(1/(_*_))

    //val eiginv = eigvalues.map(1/_)

    //val b = d * ((eigminussquare.sum)/((eiginv.sum)*(eiginv.sum)))

    print("\nThe true variance of x_1 value is\n" + sigma(1,1))

    print("\n\nThe Empirical sigma value is\n" + sigma_j(1,1))

    // plotter(amrth_sample.map((x: AM_state) => x.x), n, 0, "./exports/adaptive_trace.png")
