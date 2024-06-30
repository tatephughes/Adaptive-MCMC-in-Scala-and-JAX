val scala3Version = "3.3.1"

lazy val root = project

  .in(file("./myproject-docs"))
  .enablePlugins(MdocPlugin)

  .in(file("."))
  .settings(
    name := "Adaptive MCMC in Scala",
    version := "0.5.0",

    scalaVersion := scala3Version,

    javaOptions ++= Seq(
      "-Djava.library.path=/usr/lib/libopenblas.so"
    ),

    libraryDependencies  ++= Seq(
      "org.scalatest" %% "scalatest" % "3.2.14" % "test",
      "org.scalanlp" %% "breeze" % "2.1.0",
      "org.scalanlp" %% "breeze-viz" % "2.1.0",
      "org.scalanlp" %% "breeze-natives" % "2.1.0",
      "dev.ludovic.netlib" % "blas" % "3.0.3" withSources()
    )
  )
