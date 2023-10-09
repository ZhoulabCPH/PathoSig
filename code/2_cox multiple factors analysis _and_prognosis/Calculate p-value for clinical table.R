observed <- matrix(c(73, 19, 18, 123, 25, 28), nrow = 2, byrow = TRUE)
result <-chisq.test(observed)
p_value <- result$p.value
p_value
data <- matrix(c(137, 31, 35, 59,13, 11),
               nrow = 2, ncol = 3, byrow = TRUE)

result <-chisq.test(data)
p_value <- result$p.value
p_value

wilcox.test()
result <- fisher.test(data)
p_value <- result$p.value
p_value
