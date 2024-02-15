import org.scalatest._
import flatspec._

import breeze.plot._
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import AdaptiveMetropolis._

class AMSuite extends AnyFlatSpec{

  import AdaptiveMetropolis._

  val seed: Long = 42L

  val d = 25
  val data = Gaussian(0,1).sample(d*d).toArray.grouped(d).toArray
  val M = DenseMatrix(data: _*)
  val sigma = M.t * M
  val qr.QR(q,r) = qr(sigma)

  val state0 = AM_state(0.0,
    DenseVector.zeros[Double](d),
    DenseMatrix.eye[Double](d),
    DenseVector.zeros[Double](d)
  )

  "one_AMRTH_step" should "make a small initial step" in {
    val state1 = one_AMRTH_step(state0, r, q, seed)
    assert(norm(state1.x - state0.x) < 100*d)
  }

}
