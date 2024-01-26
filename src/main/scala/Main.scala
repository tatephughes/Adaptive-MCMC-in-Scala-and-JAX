import breeze.plot._
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import java.util.concurrent.ThreadLocalRandom



case class AM_state(j: Int, 
                    x_sum: DenseVector[Double], 
                    xxt_sum: DenseMatrix[Double],
                    x: DenseVector[Double])



object MyProgram:

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

  
  def one_AMRTH_step(state: AM_state, sigma_inv: DenseMatrix[Double]): AM_state = {

    def rng = ThreadLocalRandom.current()


    val j = state.j
    val x_sum = state.x_sum
    val xxt_sum = state.xxt_sum
    val x = state._4

    val d = x.length

    if (j <= 2*d) then { // procedure for n<=2d

      val proposed_move = MultivariateGaussian(x, DenseMatrix.eye[Double](d) * ((0.01)/d.toDouble)).draw()
      val log_acceptance_prob = math.min(0.0, 0.5 * ((x.t * sigma_inv * x) - (proposed_move.t * sigma_inv * proposed_move)))
      val u = rng.nextDouble()

      if (math.log(u) < log_acceptance_prob) then {
        return(AM_state(j+1, x_sum + proposed_move, xxt_sum + (proposed_move * proposed_move.t), proposed_move))
      }  else {
        return(AM_state(j+1, x_sum + x, xxt_sum + (x * x.t), x))
      }

    } else { // the actually adaptive part

      val sigma_j = (xxt_sum * (1/j.toDouble))
      - ((x_sum * x_sum.t) * (1/(j*j).toDouble))

      val proposed_move = (0.95) * MultivariateGaussian(x, sigma_j * (2.38*2.38/d.toDouble)).draw() + 0.05 * MultivariateGaussian(x, DenseMatrix.eye[Double](d) * ((0.01)/d.toDouble)).draw()

      val log_acceptance_prob = math.min(0.0, 0.5 * ((x.t * sigma_inv * x)
        - (proposed_move.t * sigma_inv * proposed_move)))
      val u = rng.nextDouble()

      if (math.log(u) < log_acceptance_prob) then {
        return(AM_state(j+1, x_sum + proposed_move,  xxt_sum + (proposed_move * proposed_move.t), proposed_move))
      }  else {
        return(AM_state(j+1, x_sum + x, xxt_sum + (x * x.t), x))
      }
    }

  }

  def AMRTH(state0: AM_state, sigma_inv: DenseMatrix[Double]): LazyList[AM_state] = {
    LazyList.iterate(state0)((state: AM_state) => one_AMRTH_step(state, sigma_inv))
  }


  @main def run(): Unit =
    
    val d = 100

    val data = Gaussian(0,1).sample(d*d).toArray.grouped(d).toArray
  
    val M = DenseMatrix(data: _*)
    val Sigma = M.t * M

    val state0 = AM_state(0, DenseVector.zeros[Double](d), DenseMatrix.eye[Double](d), DenseVector.zeros[Double](d))

    val amrth_sample = AMRTH(state0, inv(Sigma))

    val n = 10000
    // anything above this gives a heap space error; gonna need to optimise a bit, this is clearly innefficient; likely to do with storing a big matrix and vector along with the state, or with the way this is compututed without use of QR or anythin

    val xxt_sum = amrth_sample(n).xxt_sum
    val x_sum = amrth_sample(n).x_sum
    
    val sigma_j = (xxt_sum * (1/n.toDouble)) - ((x_sum * x_sum.t) * (1/(n*n).toDouble))

    print("\nThe true variance of x_1 value is\n" + Sigma(0,0))

    print("\n\nThe Empirical sigma value is\n" + sigma_j(0,0))

    plotter(amrth_sample.map((x: AM_state) => x.x), n, 0, "./exports/adaptive_trace.png")
    // looks like n=10000 just isn't enough for the adaptiveness to fully kick in 
