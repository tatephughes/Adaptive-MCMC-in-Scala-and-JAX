library(MASS)
library(ggplot2)
library(mvnfast)

try_accept <- function(state, prop, alpha, mix){

  j        = state$j
  x        = state$x
  x_mean   = state$x_mean
  prop_cov = state$prop_cov
  accept_count = state$accept_count
  d        = length(x)

  log_prob = min(0.0, alpha)

  u <- runif(1)

  if (log(u) < log_prob){
    x_new <- prop
    is_accepted <- 1
  } else {
    x_new <- x
    is_accepted <- 0
  }

  x_mean_new <- (x_mean*j + x_new)/(j+1)

  if (mix | j < 2*d) {
                                        # without the bias
    prop_cov_new <- prop_cov*((j-1)/j) +
      (j*tcrossprod(x_mean-x_mean_new, x_mean-x_mean_new) +
       tcrossprod(x_new - x_mean_new, x_new - x_mean_new)
      )*5.6644/(j*d)
  } else {
                                        # with the bias
    prop_cov_new <- prop_cov*((j-1)/j) +
      (j*tcrossprod(x_mean-x_mean_new, x_mean-x_mean_new) +
       tcrossprod(x_new - x_mean_new, x_new - x_mean_new) +
       0.01*diag(d)
      )*5.6644/(j*d)
  }
  
  return(list(j = j + 1,
              x = x_new,
              x_mean = x_mean_new,
              prop_cov = prop_cov_new,
              accept_count = accept_count + is_accepted))
}

adapt_step <- function(state, q, r, mix){

  j        = state$j
  x        = state$x
  prop_cov = state$prop_cov
  d        = length(x)

  if (j <= 2*d || (mix && (runif(1) < 0.01))) {
    prop <- rnorm(d)/sqrt(100*d) + x
  } else {
    prop <- rmvn(1, x, prop_cov)[1,]
  }
  
  # Compute the log acceptance probability
  alpha = 0.5 * (t(x) %*% (backsolve(r, t(q) %*% x)) - (t(prop) %*% backsolve(r, t(q) %*% prop)))
  
  return(try_accept(state, prop, alpha, mix))
}

thinned_step <- function(thinrate, state, q, r, mix){
  for (i in 1:thinrate) {
    state <- adapt_step(state, q, r, mix)
  }
  return(state)
}

sub_optim_factor <- function(sigma, sigma_j){

  # The paper suggests using these evals
  #lam = eigen(mat_sqrt(sigma_j) %*% mat_sqrt(solve(sigma)))$values

  # but the paper's code suggests this
  lam = eigen(sigma_j %*% solve(sigma))$values

  b = (length(lam) * (sum(lam^(-2)) / (sum(lam^(-1)))^2))

  return(b)
}

Rrat<-function(x)
{
  eigs<-eigen(x)$values
  sum(eigs^(-2))*length(eigs)/(sum(eigs^(-1))^2)
}

mhead <- function(M, n=5)
{
  M[0:n,0:n]
}

trace_plot <- function(sample, filepath, d){
  
  y <- sapply(sample, function(i){i$x[d]})

  df <- data.frame(index = seq_along(y), value = y)

  trace_plot <- ggplot(df, aes(x = index, y = value)) +
    geom_line(col = "#00ABFD") +
    ylab("First Coordinate Value") +
    xlab("Step") +
    labs(title = "Trace plot of the first coordinate in R")

  ggsave(filepath, plot = trace_plot, width = 590/96, height = 370/96, dpi = 96)
}

run_with_complexity <- function(sigma_d, mix){
  
  qr <- qr(sigma_d)
  Q <- qr.Q(qr)
  R <- qr.R(qr) # take the QR decomposition of sigma

  d = sqrt(length(sigma_d))
  
  n = 1
  thinrate = 1
  burnin = 1000000
  
  state <- list(j = 2, x = rep(0,d), x_mean = rep(0,d), prop_cov = (0.1)^2*diag(d)/d, 0)
  
  sample <- vector("list", n)

  start_time = Sys.time()
  
  # burnin
  for (i in 1:burnin) {
    state <- adapt_step(state, Q, R, mix)
  }

  # after burnin
  for (i in 1:n) {
    state <- thinned_step(thinrate, state, Q, R, mix)
    sample[[i]] <- state
  }

  end_time <- Sys.time()
  duration <- as.numeric(difftime(end_time, start_time, units="secs"))
  
  sigma_j <- sample[[n]]$prop_cov

  b = sub_optim_factor(sigma_d ,sigma_j)
  
  return(c(n, thinrate, burnin, duration, b))
}

compute_time_graph <- function(sigma, mix=FALSE, csv_file = "./data/R_compute_times_test.csv"){

  d = dim(sigma)[1]
  
  y <- matrix(rep(0, 5*d), ncol=5)
  
  for (i in 1:d) {

    y[i, ] <-run_with_complexity(sigma[1:i,1:i], mix)

    print(i)
    
  }

  write.table(y, csv_file, sep = ",", col.names = FALSE, row.names = FALSE)

}

generate_sigma <- function(d) {

  M <- matrix(rnorm(d^2), nrow = d)
  sigma <- solve(t(M) %*% M) 

  return(sigma)
}

read_sigma <- function(d) {

  sigma <- as.matrix(read.csv("./data/very_chaotic_variance.csv", header = FALSE))[1:d,1:d]  

  return(sigma[1:d,1:d])
  
}

main <- function(d=10, n=1000, thinrate=1000, burnin=0,
                 mix=FALSE,
                 write_files = FALSE, # whether to write out to files
                 trace_file="./Figures/trace_plot.png",
                 sample_file = "./data/r_sample",
                 get_sigma = read_sigma,
                 prog=FALSE,
                 seed=1){

  set.seed(seed)
  
  numits <- n*thinrate + burnin

  sigma <- get_sigma(d)
  
  qr <- qr(sigma)
  Q <- qr.Q(qr)
  R <- qr.R(qr)

  state <- list(j = 1, x = rep(0,d),
                x_mean = rep(0,d),
                prop_cov = (0.1)^2*diag(d)/d,
                accept_count = 0)

  sample <- vector("list", n)

  start_time <- Sys.time()

  # burn-in period
  for (i in 1:burnin) {
    state <- adapt_step(state, Q, R, mix)
    if (prog && (i %% 1000 == 0)) {
      print(paste("burnin phase", 100*i/burnin, "% complete"))
    }
  }
  
  # main sampling period
  for (i in 1:n) {
    state <- thinned_step(thinrate, state, Q, R, mix)
    sample[[i]] <- state
    if (prog && (i %% 1000 == 0)) {
      print(paste("main phase", 100*i/n, "% complete"))
    }
  }
  
  end_time <- Sys.time()
  duration <- difftime(end_time, start_time, units="secs")

  sigma_j <- sample[[n]]$prop_cov / (5.6644/d)
  acc_rate <- sample[[n]]$accept_count / (n*thinrate + burnin)
  
  b1 <- sub_optim_factor(sigma, diag(d))
  b2 <- sub_optim_factor(sigma ,sigma_j)

  print(paste("The optimal sampling value of x_1 is", sigma[1,1] * (5.6644/d)))
  print(paste("The actual sampling value of x_1 is", sigma_j[1,1] * (5.6644/d)))
  print(paste("The initial b value is", b1))
  print(paste("The final b value is", b2))
  print(paste("The acceptance rate is", acc_rate))
  print(paste("The computation took", as.numeric(duration), "seconds"))

  if (write_files) {

    print("Computing the vector of b values...")
    b_values <- toString(lapply(lapply(sample, function(y){y$prop_cov}),
                                function(y){Re(sub_optim_factor(sigma, y))}))
    print("Done!")
    
    print(paste("Saving to the file", sample_file, "..."))

    if (mix) {
      instance = "MD"
    } else {
      instance = "IC"
    }

    samplestring = paste(sapply(sample, function(y){toString(y$x)}), collapse=', ')

    results_func = paste("output_results <- function(){",
                            "chain_r_{instance} <- mcmc(sample_r_", instance, "thin=", thinrate, "start=0); min_ess <- min(effectiveSize(chain_r_", instance, ")); print(paste('The optimal sampling value of x_1 is', ", sigma[1,1], " * (5.6644/", d, "))); print(paste('The actual sampling value of x_1 is', ", sigma_j[0,0], " * (5.6644/", d, "))); print(paste('The initial b value is', b1_r_", instance, ")); print(paste('The final b value is', b_vals_r_", instance, "[-1])); print(paste('The acceptance rate is', acc_rate_r_", instance, ")); print(paste('The computation took', compute_time_r_ ", instance, ", 'seconds')); print(paste('The minimum Effective Sample Size is', min_ess)); print(paste('The minimum ESS per second is', min_ess/compute_time_r_", instance, "}))",
                            "}")
    
    lines = c(
      "library(coda)",
      paste("b1_r", instance, " <- ", b1, sep=''),
      paste("acc_rate_r_", instance, " <- ", acc_rate, sep=''),
      paste("compute_time_r_", instance, " <- ", as.numeric(duration), sep=''),
      paste("sample_r_", instance, " <- matrix(c(", samplestring, "), ncol=", d, ", byrow=TRUE)", sep=''),
      paste("bvals_r_", instance, " <- c(", b_values, ")", sep=''),
      results_func
    )

    cat(paste(lines, collapse = "\n\n"), file = sample_file)

    print("Done!")

            # Plotting has been moved over to be external, see diagnostics.org
    # Plot the trace
    #trace_plot(sample, trace_file, 1) 
  }
  
  return(sample)
}

checkwd <- function() {
  
  # This code checks wether the working directory is correct, and if not, attemps
  # to change it.
  if (!grepl(".*/Adaptive-MCMC-in-Scala-and-JAX$", getwd(), ignore.case = TRUE)) {
    setwd("../../../")
    if (!grepl(".*/Adaptive-MCMC-in-Scala-and-JAX$", getwd(), ignore.case = TRUE)) {
      print("ERROR: Cannot find correct working directory")
    } else {
      print("Succesfully found working directory")
    }
    
  } else{
    print("In correct working directory")
  }
  
}
