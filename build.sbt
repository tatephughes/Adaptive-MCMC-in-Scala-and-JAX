val scala3Version = "3.3.1"

lazy val root = project
  .in(file("."))
  .settings(
    name := "Adaptive MCMC in Scala",
    version := "0.1.0",

    scalaVersion := scala3Version,

    libraryDependencies  ++= Seq(
      "org.scalatest" %% "scalatest" % "3.2.14" % "test",
      "org.scalanlp" %% "breeze" % "2.1.0",
      "org.scalanlp" %% "breeze-viz" % "2.1.0",
      "org.scalanlp" %% "breeze-natives" % "2.1.0",
    )
  )
