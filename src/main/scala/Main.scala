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
  
  def one_AMRTH_step(state: AM_state, q: DenseMatrix[Double], r: DenseMatrix[Double]): AM_state = {

    val j = state.j
    val x_sum = state.x_sum
    val xxt_sum = state.xxt_sum
    val x = state._4

    if (j % 1000 == 0) {
      print("\n   Running: " + j + "th iteration\n")
      print("   Completed " + j/100000 + "%\n")
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

  def AMRTH(state0: AM_state, sigma: DenseMatrix[Double]): LazyList[AM_state] = {

    val qr.QR(q,r) = qr(sigma)

    LazyList.iterate(state0)((state: AM_state) => one_AMRTH_step(state, q, r))
  }

  @main def run(): Unit =

    val d = 5

    val data = Gaussian(0,1).sample(d*d).toArray.grouped(d).toArray
  
    val M = DenseMatrix(data: _*)
    val sigma = M.t * M

    val state0 = AM_state(0.0, DenseVector.zeros[Double](d), DenseMatrix.eye[Double](d), DenseVector.zeros[Double](d))

    val n: Int = 1000000

    val amrth_sample = thin(AMRTH(state0, sigma).map(_.x).drop(1000),10).take(n)

    val sigma_j = cov(DenseMatrix(amrth_sample: _*))

    //val eigvalues = eig(sqrtm(sigma_j)*sqrtm(inv(sigma))).eigenvalues

    //val eigminussquare = eigvalues.map(1/(_*_))

    //val eiginv = eigvalues.map(1/_)

    //val b = d * ((eigminussquare.sum)/((eiginv.sum)*(eiginv.sum)))

    print("\nThe true variance of x_1 value is\n" + sigma(1,1))

    print("\n\nThe Empirical sigma value is\n" + sigma_j(1,1))

    // plotter(amrth_sample.map((x: AM_state) => x.x), n, 0, "./exports/adaptive_trace.png")
