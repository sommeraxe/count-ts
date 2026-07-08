library(AER)
library(statmod)
library(caret)
library(zoo)
library(glarma)
library(glmnet) 
library(caret) 
library(car)
library(Metrics) 
library(MASS) 
library(forecast)
library(mice)

data <- read.csv("MumbaiGLM.csv")

offset <- log(data$Population)

data=data[,-c(2,4,5,6,8,9,10,12)]

# imputation
data$Vivax <- round(na.approx(data$Vivax))

data$Date = as.Date(data$Date, format = "%d/%m/%Y")
data$month_num <- format(data$Date, "%m")
data$month_name <- months(data$Date)

# dummy variables
data$month_name <- factor(
  data$month_name,
  levels = c(
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
  ),
  ordered = TRUE
)

month_dummies <- model.matrix(~ month_name - 1, data)

colnames(month_dummies) <- levels(data$month_name)

month_dummies <- month_dummies[, !(colnames(month_dummies) %in% "January")]


# time for trend
data$time <- 1:96


data <- cbind(data, month_dummies)  

data[,c(3,4,5,8)] <- as.data.frame(scale(data[,c(3,4,5,8)]))


#####################################
set.seed(42)
# GLM with Poisson distribution
model_poiss <- glm(Vivax ~ .- Date -month_name - month_num, data = data, family = poisson())
summary(model_poiss)

# check for overdispersion: residual deviance / degrees of freedom
overdispersion_value <- model_poiss$deviance / model_poiss$df.residual
print(overdispersion_value)

dispersiontest(model_poiss)

#####################################
set.seed(42)
model_nb_vivax <- glm.nb(Vivax ~ .- Date -month_name - month_num, data = data)
plot(fitted(model_nb_vivax), residuals(model_nb_vivax, type = "deviance"), 
     xlab = "Fitted Values", ylab = "Deviance Residuals", 
     main = "Residuals vs Fitted (Negative binomial)")
abline(h = 0, col = "red", lwd = 2)

p_value_gof <- 1 - pchisq(model_nb_vivax$deviance, model_nb_vivax$df.residual)
print(paste("Goodness-of-Fit p-value:", round(p_value_gof, 4)))

pred_poiss = predict(model_nb_vivax, type = "response")
plot(data$Vivax, type="l", main = "Negative binomial regression", xlab = "Time (in months)", 
     ylab = "Cases")
lines(pred_poiss, col="red", lwd=2)
legend("topleft", 
       legend = c("Actual", "Predicted"), 
       col = c("black", "red"), 
       lty = 1, 
       lwd = 2,
       bty = "n")

Metrics::rmse(data$Vivax,predict(model_nb_vivax, type = "response"))
Metrics::mape(data$Vivax,predict(model_nb_vivax, type = "response"))

summary(model_poiss)
summary(model_nb_vivax)

#####################################
set.seed(42)
# prepare matrix
X <- model.matrix(~. - Date - month_name - month_num - Vivax, data = data)   
y <- data$Vivax
pre_model <- glm.nb(y ~ X - 1 + offset(offset)) 

# starting coefficients
start_betas <- coef(pre_model)

# starting dispersion parameter
start_alpha <- 1 / pre_model$theta

nb_glarma <- glarma(y = y, X = X, offset = offset,
                 type = "NegBin",          
                 residuals = "Pearson",    
                 phiLags = c(1, 2),           
                 thetaLags = NULL,
                 method = "FS",            
                 maxit = 1000)

plot_residuals <- function(predicted, rqr, title) {
  
  # convert to uniform [0,1]
  u_res <- pnorm(rqr)
  
  # rank transform the predictions
  rank_pred <- rank(predicted) / length(predicted)
  
  # identify the extreme outliers (points stuck at 0 or 1)
  is_outlier <- u_res <= 0.001 | u_res >= 0.999
  

  par(mfrow = c(1, 2), oma = c(0, 0, 2, 0))
  
  # ==========================================
  # LEFT: Uniform QQ Plot
  # ==========================================
  # calculate expected uniform distribution (a straight line from 0 to 1)
  expected <- ppoints(length(u_res))
  observed <- sort(u_res)
  
  plot(expected, observed, main = "QQ plot residuals",
       xlab = "Expected", ylab = "Observed",
       pch = 2, col = "black", xlim = c(0, 1), ylim = c(0, 1))
  
  abline(0, 1, col = "red", lwd = 1.5)
  
  # add the floating KS test text
  ks_p <- ks.test(u_res, "punif")$p.value
  chisq_stat <- sum(rqr^2)
  disp_p <- pchisq(chisq_stat, df = length(rqr) - 1, lower.tail = FALSE)
  
  # print KS Test text
  legend("topleft", bty = "n", text.col = "red", cex = 0.9,
         legend = c(paste("KS test: p =", round(ks_p, 4)), 
                    ifelse(ks_p < 0.05, "Deviation significant", "Deviation n.s.")))
  
  # print dispersion test
  legend("left", bty = "n", text.col = "red", cex = 0.9,
         legend = c(paste("Dispersion test: p =", round(disp_p, 4)), 
                    ifelse(disp_p < 0.05, "Deviation significant", "Deviation n.s.")))
  
  # ==========================================
  # RIGHT: Scatter Plot
  # ==========================================
  # plot the normal points first
  plot(rank_pred[!is_outlier], u_res[!is_outlier], 
       main = "", xlab = "Model predictions (rank transformed)", 
       ylab = "Uniform Residual",
       pch = 1, col = "black", ylim = c(0, 1), xlim = c(0, 1))
  
  # overlay the outliers
  points(rank_pred[is_outlier], u_res[is_outlier], pch = 8, col = "red")
  
  mtext("Residual vs. predicted\nOutliers detected (red asterisks)", 
        side = 3, line = 0.5, col = "red", font = 2, cex = 0.8)
  
  # add expected quantile dashed lines
  abline(h = c(0.25, 0.5, 0.75), lty = 2, col = "gray50")
  
  # add a curving red trend line to show data deviation
  smooth <- suppressWarnings(loess(u_res ~ rank_pred))
  ord <- order(rank_pred)
  lines(rank_pred[ord], smooth$fitted[ord], col = "red", lwd = 2)
  
  # add the main title at the very top
  mtext(title, outer = TRUE, font = 2, cex = 1.2)
  
  par(mfrow = c(1, 1), oma = c(0, 0, 0, 0))
}

set.seed(42)
plot_residuals(predict(model_poiss, type = "response"), 
                          qresiduals(model_poiss), 
                          "Poisson")

set.seed(42)
plot_residuals(predict(model_nb_vivax, type = "response"), 
                          qresiduals(model_nb_vivax), 
                          "Negative binomial")


#####################################
temporal_diagnostics <- function(rqr, time = NULL) {
  
  # transform exact normal to uniform [0, 1]
  u_res <- pnorm(rqr)
  
  # handle the time index
  if (is.null(time)) {
    time <- 1:length(u_res)
  }
  
  # perform Durbin-Watson Test for temporal autocorrelation
  dw_test <- lmtest::dwtest(u_res ~ 1)
  dw_p <- dw_test$p.value
  
  # set up the 1x2 plot layout
  par(mfrow = c(1, 2), oma = c(0, 0, 2, 0))
  
  # ==========================================
  # LEFT: Residuals vs. Time
  # ==========================================
  plot(time, u_res, type = "l", 
       main = "Residuals vs. time",
       xlab = "Time", ylab = "Scaled residuals", 
       ylim = c(0, 1), col = "black")
  
  abline(h = c(0, 0.25, 0.75, 1), lty = 2, col = "black")
  abline(h = 0.5, lty = 1, col = "black") # Solid median line
  
  # ==========================================
  # RIGHT: Autocorrelation (ACF)
  # ==========================================
  # calculate ACF
  acf_res <- acf(u_res, plot = FALSE, na.action = na.pass)
  
  # plot the ACF spikes
  plot(acf_res, main = "Autocorrelation", 
       xlab = "Lag", ylab = "ACF", 
       ylim = c(-1, 1))
  
  # add the Durbin-Watson test text to the top right
  legend("topright", bty = "n", cex = 0.9,
         legend = c(paste("Durbin-Watson test p =", round(dw_p, 5)), 
                    ifelse(dw_p < 0.05, "Deviation significant", "Deviation n.s.")))
  
  par(mfrow = c(1, 1), oma = c(0, 0, 0, 0))
}


set.seed(42)
temporal_diagnostics(qresiduals(model_nb_vivax))

set.seed(42)
temporal_diagnostics(residuals(nb_glarma, type = "PIT"))

#####################################
rolling_forecast <- function(data, horizons = c(3, 6, 9, 12), nsims = 500) {
  
  all_horizon_outputs <- list()
  
  # outer loop for each Horizon
  for (predict_ahead in horizons) {
    cat("\n========================================\n")
    cat("Starting Rolling Forecast for Horizon:", predict_ahead, "\n")
    cat("========================================\n")
    
    results <- data.frame(MAPE1=numeric(), MAPE2=numeric(), MAPE3=numeric(), MAPE4=numeric(),
                          RMSE1=numeric(), RMSE2=numeric(), RMSE3=numeric(), RMSE4=numeric())
    residuals <- list()
    predictions <- list()
    
    end <- length(data$Vivax)
    limit <- end - predict_ahead - 71
    v <- 1:limit
    
    for (i in 1:limit) {
      train_end_month <- i + 71
      
      train <- data[1:train_end_month, ]
      
      test <- data[(train_end_month + 1):(train_end_month + predict_ahead), ]
      test_actuals <- test 
      
      offset_train <- offset[1:train_end_month]
      offset_test <- offset[(train_end_month + 1):(train_end_month + predict_ahead)]
      
      # Model 1: Standard Linear Model
      model1 <- lm(Vivax ~ . - Date - month_name - month_num, data = train)
      arima_model1 <- forecast::Arima(model1$residuals, order = c(1, 1, 0))
      arima_model1_preds <- forecast::forecast(arima_model1, predict_ahead)$mean
      predictions1 <- predict(model1, newdata = test) + arima_model1_preds
      
      # Model 2: Log-Transformed Linear Model
      model2 <- lm(log(Vivax) ~ . - Date - month_name - month_num, data = train)
      log_predictions <- predict(model2, newdata = test)
      v_squared <- summary(model2)$sigma^2
      v[i] <- v_squared
      arima_model2 <- forecast::Arima(model2$residuals, order = c(1, 1, 0))
      arima_model2_preds <- forecast::forecast(arima_model2, predict_ahead)$mean
      predictions2 <- exp(log_predictions + (1/2 * v_squared) + arima_model2_preds)
      
      # prepare Matrices for GLARMA
      X_train <- model.matrix(~ . - Date - month_name - month_num - Vivax, data = train)   
      X_test <- model.matrix(~ . - Date - month_name - month_num - Vivax, data = test)   
      y_train <- train$Vivax
      
      # Model 3: GLARMA Poisson
      predictions3 <- 0
      model3 <- glarma(y = y_train, X = X_train, offset = offset_train,
                       type = "Poi",          
                       residuals = "Pearson",    
                       phiLags = c(1, 2),         
                       thetaLags = NULL,        
                       method = "FS",       
                       maxit = 1000)
      
      # matrix to store Poisson mu simulations
      sim_matrix_mu_poi <- matrix(0, nrow = nsims, ncol = predict_ahead)
      
      for (j in 1:nsims) {
        sim_matrix_mu_poi[j, ] <- glarma::forecast(model3, predict_ahead, newdata = X_test, newoffset = offset_test)$mu
      }
      
      # calculate median across the 500 simulations
      predictions3 <- apply(sim_matrix_mu_poi, 2, median)
      
      # Model 4: GLARMA Negative Binomial
      model4 <- glarma(y = y_train, X = X_train, offset = offset_train,
                       type = "NegBin",          
                       residuals = "Pearson",    
                       phiLags = c(1,2),           
                       thetaLags = NULL,         
                       method = "FS",           
                       maxit = 1000)
      
      # matrix to store the simulated underlying means (mu)
      sim_matrix_mu <- matrix(0, nrow = nsims, ncol = predict_ahead)
      
      for (j in 1:nsims) {
        sim_matrix_mu[j, ] <- glarma::forecast(model4, predict_ahead, newdata = X_test, newoffset = offset_test)$mu
      }
      
      predictions4 <- apply(sim_matrix_mu, 2, median)
      
      models <- list(model1, model2, model3, model4)
      residuals[[i]] <- list(resid(model1), resid(model2), resid(model3), resid(model4))
      predictions[[i]] <- list(predictions1, log_predictions, predictions3, predictions4)
      
      # calculate MAPEs against test
      mape1 <- mean(abs((test_actuals$Vivax - predictions1) / test_actuals$Vivax))
      mape2 <- mean(abs((test_actuals$Vivax - predictions2) / test_actuals$Vivax))
      mape3 <- mean(abs((test_actuals$Vivax - predictions3) / test_actuals$Vivax))
      mape4 <- mean(abs((test_actuals$Vivax - predictions4) / test_actuals$Vivax))
      
      # calculate RMSEs against test
      rmse1 <- sqrt(mean((test_actuals$Vivax - predictions1)^2))
      rmse2 <- sqrt(mean((test_actuals$Vivax - predictions2)^2))
      rmse3 <- sqrt(mean((test_actuals$Vivax - predictions3)^2))
      rmse4 <- sqrt(mean((test_actuals$Vivax - predictions4)^2))
      
      cat(sprintf("   Completed fold: %d (Train End: %d)\n", i, train_end_month))
      results[i, 1:8] <- c(mape1, mape2, mape3, mape4, rmse1, rmse2, rmse3, rmse4)
    }
    
    all_horizon_outputs[[paste0("Horizon_", predict_ahead)]] <- list(
      Results = results, 
      Predictions = predictions, 
      Residuals = residuals
    )
  }
  
  cat("\nAll horizons completed successfully!\n")
  return(all_horizon_outputs)
}

set.seed(42)
final_output <- rolling_forecast(
    data = data, 
    horizons = c(3, 6, 9, 12)
)

mean(final_output$Horizon_3$Results$MAPE1)
mean(final_output$Horizon_3$Results$MAPE2)
mean(final_output$Horizon_3$Results$MAPE3)
mean(final_output$Horizon_3$Results$MAPE4)

mean(final_output$Horizon_6$Results$MAPE1)
mean(final_output$Horizon_6$Results$MAPE2)
mean(final_output$Horizon_6$Results$MAPE3)
mean(final_output$Horizon_6$Results$MAPE4)

mean(final_output$Horizon_9$Results$MAPE1)
mean(final_output$Horizon_9$Results$MAPE2)
mean(final_output$Horizon_9$Results$MAPE3)
mean(final_output$Horizon_9$Results$MAPE4)

mean(final_output$Horizon_12$Results$MAPE1)
mean(final_output$Horizon_12$Results$MAPE2)
mean(final_output$Horizon_12$Results$MAPE3)
mean(final_output$Horizon_12$Results$MAPE4)


horizons <- c(3, 6, 9, 12)
model_names <- c("LM + ARIMA", "Log-LM + ARIMA", "GLARMA Poisson", "GLARMA NegBin")

mape_combined_table <- matrix(NA, nrow = length(horizons), ncol = length(model_names))
rmse_combined_table <- matrix(NA, nrow = length(horizons), ncol = length(model_names))

rownames(mape_combined_table) <- rownames(rmse_combined_table) <- paste0("Horizon_", horizons)
colnames(mape_combined_table) <- colnames(rmse_combined_table) <- model_names

for (i in seq_along(horizons)) {
  h_name <- paste0("Horizon_", horizons[i])
  res_df <- final_output[[h_name]]$Results
  
  mape_data <- res_df[, 1:4]
  mape_means <- colMeans(mape_data, na.rm = TRUE)
  mape_sds <- apply(mape_data, 2, sd, na.rm = TRUE)
  
  mape_combined_table[i, ] <- sprintf("%.4f ± %.4f", mape_means, mape_sds)
  
  rmse_data <- res_df[, 5:8]
  rmse_means <- colMeans(rmse_data, na.rm = TRUE)
  rmse_sds <- apply(rmse_data, 2, sd, na.rm = TRUE)
  
  rmse_combined_table[i, ] <- sprintf("%.2f ± %.2f", rmse_means, rmse_sds)
}

cat("=== MAPE Stability: Mean ± SD (Percentage Error) ===\n")
print(mape_combined_table, quote = FALSE)

cat("\n=== RMSE Stability: Mean ± SD (Raw Vivax Cases) ===\n")
print(rmse_combined_table, quote = FALSE)

par(mfrow=c(1,2))
# ---------------------------------------------------------
# PLOT 1: Average MAPE vs. Forecast Horizon
# ---------------------------------------------------------
matplot(x = horizons, y = avg_mape_table, type = "b", 
        pch = 15:18, col = c("black", "blue", "red", "darkgreen"), lty = 1, lwd = 2,
        xlab = "Forecast Horizon (Months Ahead)", 
        ylab = "Average MAPE (%)",
        main = "Accuracy Degradation (MAPE)", 
        xaxt = "n")

axis(1, at = horizons)
legend("topleft", legend = model_names, 
       col = c("black", "blue", "red", "darkgreen"), 
       pch = 15:18, lty = 1, lwd = 2, bty = "n", cex = 1, text.font = 2)

# ---------------------------------------------------------
# PLOT 2: Average RMSE vs. Forecast Horizon
# ---------------------------------------------------------
matplot(x = horizons, y = avg_rmse_table, type = "b", 
        pch = 15:18, col = c("black", "blue", "red", "darkgreen"), lty = 1, lwd = 2,
        xlab = "Forecast Horizon (Months Ahead)", 
        ylab = "Average RMSE (Case Count)",
        main = "Error Magnitude (RMSE)", 
        xaxt = "n")

axis(1, at = horizons)

par(mfrow = c(1, 1))

par(mfrow = c(2, 2), mar = c(6, 4, 3, 1)) 

box_colors <- c("gray90", "lightblue", "pink", "lightgreen")

for (h in horizons) {
  h_name <- paste0("Horizon_", h)
  
  res_mape <- final_output[[h_name]]$Results[, 1:4]
  colnames(res_mape) <- c("LM+AR", "Log+AR", "Poi", "NB")
  
  # generate the boxplot
  boxplot(res_mape, 
          col = box_colors,
          main = paste("Stability at Horizon", h),
          ylab = "MAPE per Fold",
          las = 2,      
          cex.axis = 0.9, 
          outpch = 19,    
          outcol = "red") 

par(mfrow = c(1, 1))
