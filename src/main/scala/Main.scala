import breeze.plot._
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister
import java.awt.Color
import java.io.PrintWriter
import reflect.Selectable.reflectiveSelectable
import scala.io.Source

implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(41L)))

case class AM_state(j: Double, 
                    x: DenseVector[Double], 
                    x_mean: DenseVector[Double],
                    prop_cov: DenseMatrix[Double])

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

    p.title = "Trace Plot of the first coordinate, d=" + sample(1).length
    
    f.saveas(file_path)

  }
  
  def adapt_step(state: AM_state, q: dm, r: dm, prog: Boolean): AM_state = {

    val j = state.j
    val x = state.x
    val x_mean = state.x_mean
    val prop_cov = state.prop_cov

    // print progress every 1000 iterations if 'prog=true'
    if (prog && j % 1000 == 0) {
      print("\n   Running: " + j.toInt + "th iteration\n")
      val runtime = Runtime.getRuntime()
      print(s"** Used Memory (MB): ${(runtime.totalMemory-runtime.freeMemory)/(1048576)}")
    }

    val d = x.length
    val prop = MultivariateGaussian(x, prop_cov).draw()
    val alpha = 0.5 * ((x.t * (r \ (q.t * x))) - (prop.t * (r \ (q.t * prop))))
    val log_prob = math.min(0.0, alpha)
    val u = Uniform(0,1).draw()

    val x_new = if (math.log(u) < log_prob) then {
      prop
    } else {
      x
    }

    val x_mean_new = ((j-1)*x_mean + x_new)/j

    val prop_cov_new = if (j <= 2*d) {
      prop_cov
    } else {
      prop_cov*(j-1)/j +
        (j * (x_mean * x_mean.t) -
        (j+1)*(x_mean_new * x_mean_new.t) +
        (x_new * x_new.t) +
        0.01*(DenseMatrix.eye[Double](d))) * 5.6644/(j*d).toDouble
    }

    return(AM_state(j+1, x_new, x_mean_new, prop_cov_new))

  }

  def AM_iterator(state0: AM_state, sigma: dm, prog: Boolean): LazyList[AM_state] = {

    val qr.QR(q,r) = qr(sigma)

    LazyList.iterate(state0)((state: AM_state) => adapt_step(state, q, r, prog))
  }

  def effectiveness(sigma: dm, sigma_j: dm): Double = {

    // PROBABLY DON'T USE, not sure why but this doesn't seem to work
    // i think it works now?


    val d = sigma.cols

    val sigma_j_decomp = eig(sigma_j)
    val sigma_decomp = eig(sigma)

    val rootsigmaj = sigma_j_decomp.eigenvectors * diag(sqrt(sigma_j_decomp.eigenvalues)) * inv(sigma_j_decomp.eigenvectors)
    val rootsigmainv  = inv(sigma_j_decomp.eigenvectors * diag(sqrt(sigma_decomp.eigenvalues)) * inv(sigma_decomp.eigenvectors))

    val lambda = eig(rootsigmaj * rootsigmainv).eigenvalues
    val lambdaminus2sum = sum(lambda.map(x => 1/(x*x)))
    val lambdainvsum = sum(lambda.map(x => 1/x))

    // According to Roberts and Rosenthal, this should go to
    // 1 at the stationary distribution
    val b = d * (lambdaminus2sum / (lambdainvsum*lambdainvsum))

    return(b)

  }

  def run_with_complexity(sigma_d: dm): (Int, Int, Int, Double, Double) = {

    val d = sigma_d.cols
    val n: Int = 10000      // size of the desired sample
    val thinrate: Int = 10   // the thinning rate
    val burnin: Int = 1000000 // the number of iterations for burn-in

    val state0 = AM_state(1.0,
                          DenseVector.zeros[Double](d),
                          DenseVector.zeros[Double](d),
                          0.01 * DenseMatrix.eye[Double](d) / d.toDouble)

    // start of the computation
    val startTime = System.nanoTime()

    val am_sample = AM_iterator(state0, sigma_d, false).drop(burnin).thin(thinrate).take(n).toArray

    // the time of computation is seconds
    val endTime = System.nanoTime()
    val duration = (endTime - startTime) / 1e9d

    val sigma_j = cov(DenseMatrix(am_sample.map(_.x): _*))

    val sigma_j_decomp = eig(sigma_j)
    val sigma_decomp = eig(sigma_d)

    val rootsigmaj = sigma_j_decomp.eigenvectors * diag(sqrt(sigma_j_decomp.eigenvalues)) * inv(sigma_j_decomp.eigenvectors)
    val rootsigmainv  = inv(sigma_decomp.eigenvectors * diag(sqrt(sigma_decomp.eigenvalues)) * inv(sigma_decomp.eigenvectors))

    val lambda = eig(rootsigmaj * rootsigmainv).eigenvalues
    val lambdaminus2sum = sum(lambda.map(x => 1/(x*x)))
    val lambdainvsum = sum(lambda.map(x => 1/x))

    // According to Roberts and Rosenthal, this should go to
    // 1 at the stationary distribution
    val b = d * (lambdaminus2sum / (lambdainvsum*lambdainvsum))

    return(n,thinrate,burnin, duration, b)

  }

  def compute_time_graph(sigma: dm, csv_file: String): Unit = {

    val d = sigma.cols

    val x = 1 to d

    val y = for (i <- x) yield {print(s"\n$i"); run_with_complexity(sigma(0 to i-1, 0 to i-1))}

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
    
  def run(): Unit =

    // Read the file lines, skipping empty lines
    val lines = Source.fromFile("data/chaotic_variance.csv").getLines().filter(_.nonEmpty).toList

    // Assuming the CSV is well-formed and every row has the same number of columns
    val data = lines.map(_.split(",").map(_.toDouble))

    // Extracting the row and column counts
    val numRows = data.length
    val numCols = data.head.length

    // Creating the DenseMatrix
    val sigma_d = DenseMatrix(data: _*).reshape(numRows, numCols)

    compute_time_graph(sigma_d, "data/scala_compute_times_laptop_1.csv")

  @main def simple_run(): Unit = 

    //val d: Int = 10
    //val n: Int = 100000        // size of the desired sample
    //val thinrate: Int = 10     // the thinning rate
    //val burnin: Int = 10000    // the number of iterations for burn-in


    // high d
    val d: Int = 10
    val n: Int = 100000
    val thinrate: Int = 10
    val burnin: Int = 100000

    // initial state
    val state0 = AM_state(1.0,
                          DenseVector.zeros[Double](d),
                          DenseVector.zeros[Double](d),
                          0.01 * DenseMatrix.eye[Double](d) / d.toDouble)

    val data = Gaussian(0,1).sample(d*d).toArray.grouped(d).toArray
    val M = DenseMatrix(data: _*)
    val sigma = M.t * M

    val startTime = System.nanoTime()

    // the sample
    val am_sample = AM_iterator(state0, sigma, false).drop(burnin).thin(thinrate).take(n).toArray

    // the time of computation is seconds
    val endTime = System.nanoTime()
    val duration = (endTime - startTime) / 1e9d

    // Empirical Variance matrix of the sample
    val sigma_j = cov(DenseMatrix(am_sample.map(_.x): _*))

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

    print("\nThe true variance of x_1 is " + sigma(1,1))
    print("\nThe empirical sigma value is " + sigma_j(1,1))
    print("\nThe b value is " + b)
    print("\nThe computation took " + duration + " seconds" )

    plotter(am_sample.map(_.x), "./Figures/adaptive_trace_scala_d_10.png")
