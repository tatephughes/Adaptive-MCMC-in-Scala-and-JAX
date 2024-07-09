import breeze.plot._
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
import breeze.stats.distributions.Gaussian
import org.apache.commons.math3.random.MersenneTwister
import java.awt.Color
import java.io.PrintWriter
import reflect.Selectable.reflectiveSelectable
import scala.io.Source
import scala.util.chaining
import java.io.File
import breeze.linalg.{CSCMatrix, csvwrite}


// randombasis and seed for PRNG
implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(41L)))



// A class for the state of the chaining
// TODO add ~is_accepted~ as a element
case class AM_state(j: Double, 
                    x: DenseVector[Double], 
                    x_mean: DenseVector[Double],
                    prop_cov: DenseMatrix[Double],
                    accept_count: Int
)

extension[T](ll: LazyList[T]){
  /* Extension of the LazyList class for chain thinning, done in a memory-efficient way
   updated from Darren's scala course as an extension, github.com/darrenjw/scala-course
   */ 
  def thin(th: Int): LazyList[T] = {
    val lld = ll.drop(th - 1)
    if (lld.isEmpty) LazyList.empty else
      lld.head #:: lld.tail.thin(th)
  }
}


object AdaptiveMetropolis {

  import dev.ludovic.netlib.blas.BLAS.{ getInstance => blas }

  // Short-hand for two commonly used types
  type dm = DenseMatrix[Double]
  type dv = DenseVector[Double]

  /**
   * Backsolve an upper-triangular linear system
   * with a single RHS
   *
   * @param A An upper-triangular matrix
   * @param y A single vector RHS
   *
   * @return The solution, x, of the linear system A x = y
   *
   * Credit: Darren Wilkinson, https://github.com/darrenjw/scala-glm
   */
  def backSolve(A: dm,
    y: dv): dv = {
    val yc = y.copy
    blas.dtrsv("U", "N", "N", A.cols, A.toArray,
      A.rows, yc.data, 1)
    yc
  }

  def plot_trace(sample: Array[dv], // the sample to plot
               file_path: String  // file to save to
    ): Unit = {

    /* Plots a trace plot of the dth coordinate of the given array of states,
    and saves the figure to ~file_path~
     */

    val y = DenseVector(sample.map(x => x(0)))
    val x = DenseVector.tabulate(y.length)(i => (i+1).toDouble)

    val f = Figure("Trace Plot of the first coordinate")
    val p = f.subplot(0)

    p += plot(x,y)
    p.xlabel = "Step"
    p.ylabel = "First coordinate value"

    p.title = "Trace Plot of the first coordinate, d = " + sample(1).length
    
    f.saveas(file_path)

  }

  def try_accept(state: AM_state,
                 prop: dv,
                 alpha: Double,
                 mix: Boolean
    ): AM_state = {

      val j = state.j
      val x = state.x
      val x_mean = state.x_mean
      val prop_cov = state.prop_cov
      val accept_count = state.accept_count
      val d = x.length

      val log_prob = math.min(0.0, alpha)

      // Try accept the proposal
      val (x_new, is_accepted) = if (math.log(Uniform(0,1).draw()) < log_prob) then {
        (prop, 1)
      } else {
        (x, 0)
      }

      // update empirical mean
      val x_mean_new = (j*x_mean + x_new)/(j+1)

      // update proposal covariance
      val prop_cov_new = if (mix | j < 2*d) {
        prop_cov*((j-1)/j) +
          (j * ((x_mean-x_mean_new) * (x_mean-x_mean_new).t) +
          ((x_new-x_mean_new) * (x_new-x_mean_new).t)) * 5.6644/(j*d)
      } else {
        prop_cov*((j-1)/j) +
          (j * ((x_mean-x_mean_new) * (x_mean-x_mean_new).t) +
          ((x_new-x_mean_new) * (x_new-x_mean_new).t) +
            0.01*(DenseMatrix.eye[Double](d))) * 5.6644/(j*d)
      }

      return(AM_state(j+1, x_new, x_mean_new, prop_cov_new, accept_count+is_accepted))

    }

  def adapt_step(state: AM_state, // The state of the chain
                 q: dm, r: dm,    // The QR-decomposition of the target Covariance
                 prog: Boolean,   // Flag for whether to print diagnositcs to console
                 mix: Boolean     // Whether to use the MD version of the algorithm
    ): AM_state = {

    /* Samples from the current proposal distribution and computes the log
    Hastings Ratio, and returns the next state
     */

    // Extract values from the state
    // TODO Maybe remove to reduce cloning
    val j = state.j
    val x = state.x
    val x_mean = state.x_mean
    val prop_cov = state.prop_cov
    val accept_count = state.accept_count
    val d = x.length

    // print progress every 1000 iterations if 'prog=true'
    if (prog && j % 1000 == 0) {
      print("\n   Running: " + j.toInt + "th iteration\n")
      val runtime = Runtime.getRuntime()
      print(s"** Used Memory (MB): ${(runtime.totalMemory-runtime.freeMemory)/(1048576)}")
    }

    val prop = if (j <= 2*d | (mix & (Uniform(0,1).draw() < 0.01))){
      // 'Safe' sampler
      DenseVector(Gaussian(0,1).sample(d).toArray) / sqrt(100*d) + x
      //MultivariateGaussian(x, 1/sqrt(100*d)*DenseMatrix.eye[Double](d)).draw()
    } else {
      // 'Adaptive' sampler
      MultivariateGaussian(x, prop_cov).draw()
    }

    // The log Hastings Ratio (backSolve isn't exposed in breeze, but i seem to get much worse performance with it for some reason?)
    //val alpha = 0.5 * ((x.t * (r \ (q.t * x))) - (prop.t * (r \ (q.t * prop))))
    val alpha = 0.5 * ((x.t * backSolve(r, (q.t * x))) - (prop.t * backSolve(r, (q.t * prop))))
    //val alpha2 = 0.5 * ((x.t * backSolve(r, q.t * x))) - (prop.t * backSolve(r, q.t * prop))

    //val num1 = (x.t * (r \ (q.t * x)))
    //val num2 = (x.t * backSolve(r, q.t * x))

    //val num1 = (prop.t * (r \ (q.t * prop)))
    //val num2 = (prop.t * backSolve(r, q.t * prop))

    //println(s"\n$alpha")
    //println(s"$alpha2\n")

    return(try_accept(state, prop, alpha, mix))

  }

  def AM_iterator(state0: AM_state, // the initial state of the Chain
                  sigma: dm,        // The target Covariance matrix
                  prog: Boolean,    // Flag for whether to print diagnostics to console
                  mix: Boolean
    ): LazyList[AM_state] = {

    /* Iterates ~adapt_step~ to create a Lazy sample of arbitrary length
     */

    val qr.QR(q,r) = qr(sigma)

    LazyList.iterate(state0)((state: AM_state) => adapt_step(state, q, r, prog, mix))
  }

  def sub_optim_factor(sigma: dm, sigma_j: dm): Double = {

    /* Computes the sub-optimality factor between the true target covariance
    ~sigma~ and the sampling covariance ~sigma_j~, from Roberts and Rosethal
     */

    val lambda = eig(sigma_j * inv(sigma)).eigenvalues

    val lambdaminus2sum = sum(lambda.map(x => 1/(x*x)))
    val lambdainvsum = sum(lambda.map(x => 1/x))

    // According to Roberts and Rosenthal, this should go to
    // 1 at the stationary distribution
    val b = sigma.cols * (lambdaminus2sum / (lambdainvsum*lambdainvsum))

    return(b)

  }

  def run_with_complexity(sigma_d: dm, mix: Boolean): (Int, Int, Int, Double, Double) = {

     /* Runs the main loop on a given target Covariance, and gets the time the main
     loop took.
     */

    val d = sigma_d.cols

    // These numbers get good results up to d=100
    val n: Int = 1      // size of the desired sample
    val thinrate: Int = 1   // the thinning rate
    val burnin: Int = 1000000 // the number of iterations for burn-in

    val state0 = AM_state(2.0, // starts at "2" for safety
                          DenseVector.zeros[Double](d),
                          DenseVector.zeros[Double](d),
                          0.01 * DenseMatrix.eye[Double](d) / d.toDouble,
                          0
    )

    // Start of the computation
    val startTime = System.nanoTime()

    val am_sample = AM_iterator(state0, sigma_d, false, false).drop(burnin).thin(thinrate).take(n).toArray

    val endTime = System.nanoTime()

    // The time of computation is seconds
    val duration = (endTime - startTime) / 1e9d

    val sigma_j = am_sample.last.prop_cov

    val b = sub_optim_factor(sigma_d, sigma_j)

    return(n,thinrate,burnin, duration, b)

  }

  def compute_time_graph(sigma: dm, mix: Boolean, csv_file: String = "./data/Scala_compute_times_test.csv"): Unit = {

    /* Loop through all the primary minors of ~sigma~ and runs the complexity test
    on each of them, saving the result to ~csv_file~
     */

    val d = sigma.cols

    val x = 1 to d

    val y = for (i <- x) yield {print(s"\n$i"); run_with_complexity(sigma(0 to i-1, 0 to i-1), mix)}

    using(new PrintWriter(csv_file)) { writer =>
      y.foreach { case (a, b, c, d, e) =>
        writer.println(s"$a,$b,$c,$d,$e")
      }
    }

    def using[A <: { def close(): Unit }, B](resource: A)(block: A => B): B = {
      try {
        block(resource)
      } finally {
        resource.close()
      }
    }

  }

  def generate_sigma(d:Int): dm = {

    val data = Gaussian(0,1).sample(d*d).toArray.grouped(d).toArray
    val M = DenseMatrix(data*)
    val sigma = M.t * M

    return(sigma)

  }

  def read_sigma(d: Int): dm = {

    // Read the file lines, skipping empty lines
    val lines = Source.fromFile("./data/very_chaotic_variance.csv").getLines().filter(_.nonEmpty).toList

    // Assuming the CSV is well-formed and every row has the same number of columns
    val data = lines.map(_.split(",").map(_.toDouble))

    // Extracting the row and column counts
    val numRows = data.length
    val numCols = data.head.length

    // Creating the DenseMatrix
    val sigma = DenseMatrix(data*).reshape(numRows, numCols)

    return(sigma(0 until d, 0 until d))

  }

  def main(d: Int = 10,
    n: Int = 1000, thinrate: Int = 1000,
    burnin: Int = 0,
    write_files: Boolean = false,
    trace_file: String = "./Figures/adaptive_trace_Scala.png",
    sample_file: String = "./data/scala_sample",
    prog: Boolean = false,
    mix: Boolean = false,
    get_sigma: (Int => dm) = generate_sigma): Unit = {

    // initial state
    val state0 = AM_state(2.0, // starts at "2" for safety
                          DenseVector.zeros[Double](d),
                          DenseVector.zeros[Double](d),
                          0.01 * DenseMatrix.eye[Double](d) / d.toDouble,
                          0
    )

    val startTime = System.nanoTime()

    val sigma = get_sigma(d)

    // the sample
    val sample = AM_iterator(state0, sigma, prog, mix).drop(burnin).thin(thinrate).take(n).toArray

    // the time of computation is seconds
    val endTime = System.nanoTime()
    val duration = (endTime - startTime) / 1e9d
    
    // Final Sampling covariance
    val sigma_j = sample.last.prop_cov
    val acc_rate = sample.last.accept_count.toDouble / (n*thinrate+burnin)

    // According to Roberts and Rosethal, this value should go to 1.
    val b1 = sub_optim_factor(sigma, DenseMatrix.eye(d))
    val b2 = sub_optim_factor(sigma, sigma_j)

    print("\nThe optimal sampling variance of x_1 is " + (5.6644/d) * sigma(0,0))
    print("\nThe actual sampling variance of x_1 is  " + sigma_j(0,0))
    print("\nThe initial b value is " + b1)
    print("\nThe final b value is " + b2)
    print("\nThe acceptance rate is " + acc_rate)
    print("\nThe computation took " + duration + " seconds\n" )

    if (write_files) {

      print("Computing the vector of b values...\n")
      val b_values: String = sample.map(_.prop_cov).map(sub_optim_factor(sigma, _)).toArray.mkString(", ")
      print("Done!\n")

      print(s"Saving to the file $sample_file...\n")

      val instance = if (mix) {"MD"} else {"IC"}
      val samplestring = DenseMatrix(sample.map(_.x)*).toArray.mkString(", ")

      val lines: Seq[String] = Seq(
        s"compute_time_scala_$instance <- $duration",
        s"sample_scala_$instance <- matrix(c($samplestring), ncol=$d)",
        s"bvals_scala_$instance <- c($b_values)"
      )

      val writer = new PrintWriter(new File(sample_file))
      writer.write(lines.mkString("\n\n"))
      writer.close()

      print("Done!\n")

      val x_sample = sample.map(_.x)

        // Plotting has been moved over to be external, see diagnostics.org
      // plot the trace of the first coordinate
      //plot_trace(sample.map(_.x), trace_file)
    }
  }

  @main def quick_run(): Unit = {

    main(d=3, n=1000, thinrate=1, burnin=0,
         write_files = false,
         trace_file = "./Figures/scala_trace_basetest_IC.png",
         sample_file = "./data/scala_sample_quickrun",
         get_sigma = read_sigma,
         mix = false
    )

  }

  @main def simple_run_IC(): Unit = {

    main(d=100, n=10000, thinrate=100, burnin=0,
         write_files = true,
         trace_file = "./Figures/scala_trace_basetest_IC.png",
         sample_file = "./data/scala_sample_basetest_IC",
         get_sigma = read_sigma,
         mix = false
    )
  }

  @main def simple_run_IC_quick(): Unit = {

    main(d=10, n=10000, thinrate=100, burnin=0,
         write_files = true,
         trace_file = "./Figures/scala_trace_basetest_IC_quick.png",
         sample_file = "./data/scala_sample_basetest_IC_quick",
         get_sigma = read_sigma,
         mix = false
    )
  }

  @main def simple_run_MD(): Unit = {

    main(d=100, n=10000, thinrate=100, burnin=0,
         write_files = true,
         trace_file = "./Figures/scala_trace_basetest_MD.png",
         sample_file = "./data/scala_sample_basetest_MD",
         get_sigma = read_sigma,
         mix = true
    )
  }

  @main def complexity_run_IC(): Unit = {

    compute_time_graph(read_sigma(100), false,
      "./data/scala_compute_times_laptop_1_IC.csv")

  }

  @main def complexity_run_MD(): Unit = {

    compute_time_graph(read_sigma(100), true,
      "./data/scala_compute_times_laptop_1_MD.csv")

  }
}
