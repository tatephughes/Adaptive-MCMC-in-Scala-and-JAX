import breeze.plot._
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister
import java.awt.Color

implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(42L)))


case class AM_state(j: Double, 
                    x_sum: DenseVector[Double], 
                    xxt_sum: DenseMatrix[Double],
                    x: DenseVector[Double])

extension[T](ll: LazyList[T]){
  // updated from Darren's scala course as an extension, github.com/darrenjw/scala-course
  def thin(th: Int): LazyList[T] = {
    val lld = ll.drop(th - 1)
    if (lld.isEmpty) LazyList.empty else
      lld.head #:: lld.tail.thin(th)
  }
}

object AdaptiveMetropolis:

  type dm = DenseMatrix[Double]
  type dv = DenseVector[Double]

  def plotter(sample: Array[dv], // the sample to plot
              file_path: String): Unit = {

    val y = DenseVector(sample.map(x => x(1)))
    val x = DenseVector.tabulate(y.length)(i => (i+1).toDouble)

    val f = Figure("Trace Plot of the first coordinate")
    val p = f.subplot(0)

    p += plot(x,y)
    p.xlabel = "Step"
    p.ylabel = "First coordinate value"

    p.title = "Trace Plot of the first coordinate"
    
    f.saveas(file_path)

  }
  
  def AM_step(state: AM_state, q: dm, r: dm, prog: Boolean): AM_state = {

    val j = state.j
    val x_sum = state.x_sum
    val xxt_sum = state.xxt_sum
    val x = state._4

    // print progress every 1000 iterations if 'prog=true'
    if (prog && j % 1000 == 0) {
      print("\n   Running: " + j.toInt + "th iteration\n")
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

  def AM_iterator(state0: AM_state, sigma: dm, prog: Boolean): LazyList[AM_state] = {

    val qr.QR(q,r) = qr(sigma)

    LazyList.iterate(state0)((state: AM_state) => AM_step(state, q, r, prog))
  }

  @main def run(): Unit =

    val startTime = System.nanoTime()

    val d = 10               // dimension of the state space
    val n: Int = 100000      // size of the desired sample
    val thinrate: Int = 10   // the thinning rate
    val burnin: Int = 100000 // the number of iterations for burn-in
    
    // the actual number of iterations computed is n*thin + burnin

    // create a chaotic variance matrix to target
    val data = Gaussian(0,1).sample(d*d).toArray.grouped(d).toArray
    val M = DenseMatrix(data: _*)
    val sigma = M.t * M

    // initial state
    val state0 = AM_state(0.0,
                          DenseVector.zeros[Double](d),
                          DenseMatrix.eye[Double](d),
                          DenseVector.zeros[Double](d))

    
    // the sample
    val am_sample = AM_iterator(state0, sigma, false).drop(burnin).thin(thinrate).map(_.x).take(n).toArray

    // Empirical Variance matrix of the sample
    val sigma_j = cov(DenseMatrix(am_sample: _*))

    val sigma_j_decomp = eig(sigma_j)
    val sigma_decomp = eig(sigma)

    val rootsigmaj = sigma_j_decomp.eigenvectors * diag(sqrt(sigma_j_decomp.eigenvalues)) * inv(sigma_j_decomp.eigenvectors)
    val rootsigmainv  = inv(sigma_decomp.eigenvectors * diag(sqrt(sigma_decomp.eigenvalues)) * inv(sigma_decomp.eigenvectors))

    val lambda = eig(rootsigmaj * rootsigmainv).eigenvalues
    val lambdaminus2sum = sum(lambda.map(x => 1/(x*x)))
    val lambdainvsum = sum(lambda.map(x => 1/x))

    // According to Roberts and Rosenthal, this should go to
    // 1 at the stationary distribution
    val b = d * (lambdaminus2sum / (lambdainvsum*lambdainvsum))

    // the time of computation is seconds
    val endTime = System.nanoTime()
    val duration = (endTime - startTime) / 1e9d

    print("\nThe true variance of x_1 is " + sigma(1,1))
    print("\nThe empirical sigma value is " + sigma_j(1,1))
    print("\nThe b value is " + b)
    print("\nThe computation took " + duration + " seconds" )

    plotter(am_sample, "./Figures/adaptive_trace_scala.png")
