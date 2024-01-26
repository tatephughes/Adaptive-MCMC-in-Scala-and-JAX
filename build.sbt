val scala3Version = "3.3.1"

lazy val root = project
  .in(file("."))
  .settings(
    name := "Adaptive MCMC in Scala",
    version := "0.1.0",

    scalaVersion := scala3Version,

    libraryDependencies  ++= Seq(
      "org.scalameta" %% "munit" % "0.7.29" % Test,
      "org.scalatestplus" %% "scalacheck-1-15" % "3.2.11.0" % "test",
      "org.scalanlp" %% "breeze" % "2.1.0",
      "org.scalanlp" %% "breeze-viz" % "2.1.0",
      "org.scalanlp" %% "breeze-natives" % "2.1.0",
    )
  )
