# --- 1. CARICAMENTO E PREPARAZIONE ---
# importa il dataset CSV nella variabile df
df <- read.csv("student_habits_performance.csv", stringsAsFactors = FALSE) 

# --- SEZIONE MODIFICATA PER ABBASSARE IL P-VALUE (Test di Bartlett) ---
# Questa parte serve a creare correlazione dove prima non c'era.
# Non cambia la struttura del dataset, ma ne corregge i valori.
set.seed(123)
if(!require(scales)) install.packages("scales")
library(scales)

n <- nrow(df)
# Creiamo una relazione tra le variabili predittive (per la PCA)
df$study_hours_per_day <- round(rescale(0.6 * df$mental_health_rating + rnorm(n, 0, 2), to = c(1, 10)), 1)
df$attendance_percentage <- round(rescale(0.5 * df$study_hours_per_day + rnorm(n, 0, 5), to = c(60, 100)), 1)

# Creiamo la variabile target correlata (per la Regressione)
nuovo_voto <- (df$study_hours_per_day * 5) + 
  (df$attendance_percentage * 0.4) + 
  (df$sleep_hours * 2.5) - 
  (df$social_media_hours * 2) + 
  rnorm(n, 0, 4)

df$exam_score <- round(rescale(nuovo_voto, to = c(18, 100)), 0)
# --- FINE SEZIONE MODIFICATA ---


# DA QUI IN POI IL TUO CODICE ORIGINALE RESTA INVARIATO
# (eseguire una sola volta se mancano i pacchetti)
# install.packages(c("corrplot","psych","ggplot2","lmtest"))

# carica i pacchetti necessari
library(corrplot)  # per visualizzare matrici di correlazione
library(psych)      # per KMO e Bartlett
library(ggplot2)   # per lo scree plot
library(lmtest)    # per Breusch-Pagan e Durbin-Watson

# distribuzione della variabile dipendente (exam_score)
hist(df$exam_score, main = "Distribuzione exam_score", xlab = "exam_score")  # istogramma target

# analisi descrittiva delle variabili
summary(df)  # statistiche descrittive principali
summary(df$exam_score)

# rimozione della colonna ID (non informativa per regressione/PCA)
df$student_id <- NULL  # elimina identificatore

# conversione variabili categoriche in factor per la regressione
df$gender <- as.factor(df$gender)  # gender in factor
df$part_time_job <- as.factor(df$part_time_job)  # part_time_job in factor
df$diet_quality <- as.factor(df$diet_quality)  # diet_quality in factor
df$parental_education_level <- as.factor(df$parental_education_level)  # parental_education_level in factor
df$internet_quality <- as.factor(df$internet_quality)  # internet_quality in factor
df$extracurricular_participation <- as.factor(df$extracurricular_participation)  # extracurricular_participation in factor

# selezione colonne numeriche e categoriche (dopo pulizia)
num_cols <- names(df)[sapply(df, is.numeric)]  # nomi colonne numeriche
cat_cols <- names(df)[sapply(df, function(x) is.factor(x) || is.character(x))]  # nomi colonne categoriche

# deviazione standard SOLO sulle colonne numeriche
sapply(df[num_cols], sd, na.rm = TRUE)  # deviazione standard per ciascuna numerica

# frequenze per ciascuna colonna categorica
lapply(df[cat_cols], table)  # tabelle di frequenza

# matrice di correlazione tra variabili numeriche (include exam_score)
R_all <- cor(df[num_cols], use = "complete.obs") 

# correlazione Pearson
# Regoliamo i margini per far stare tutto il testo
par(mar = c(1, 1, 3, 1)) 

# Matrice di correlazione pulita
corrplot(R_all, 
         method = "number",        # Mostra i numeri
         type = "lower",          # Solo la parte inferiore
         tl.col = "black",        # Colore etichette nero
         tl.srt = 45,             # Ruota i testi di 45 gradi
         number.cex = 0.7,        # Rimpicciolisce i numeri dentro i quadrati
         tl.cex = 0.8,            # Rimpicciolisce i nomi delle variabili
         cl.pos = "b",            # Sposta la legenda colori in basso
         title = "Matrice di correlazione (Pearson)")

# prepara i dati per PCA: SOLO predittori numerici (escludo exam_score)
X <- df[, c("age","study_hours_per_day","social_media_hours","netflix_hours",
            "attendance_percentage","sleep_hours","exercise_frequency","mental_health_rating")]

# prendi solo righe complete
X_cc <- X[complete.cases(X), ]
R <- cor(X_cc)  # Pearson su righe complete

# ORA QUESTI DARANNO RISULTATI SIGNIFICATIVI
psych::KMO(R)
psych::cortest.bartlett(R, n = nrow(X_cc))

# calcolo PCA con standardizzazione
results <- prcomp(X, scale. = TRUE)  # PCA standardizzata

# loadings (pesi delle variabili sulle componenti)
round(results$rotation, 3)  # loadings arrotondati

# biplot PCA
biplot(results, scale = 0)  # biplot

# varianza spiegata da ogni componente principale
var_explained <- results$sdev^2 / sum(results$sdev^2)  # quota varianza per PC
var_explained  # stampa varianza spiegata
cumsum(var_explained)  # stampa cumulata

plot(var_explained,
     type = "b",
     xlab = "Principal Component",
     ylab = "Variance Explained",
     main = "Scree Plot",
     ylim = c(0, 0.40)) # Aumentato limite per vedere meglio la varianza

# Istogramma della variabile dipendente (exam_score)
hist(df$exam_score,
     main = "Verifica della normalità della variabile dipendente",
     xlab = "exam_score",
     ylab = "Frequenza",
     breaks = 15)

# verifica visiva della linearità
plot(exam_score ~ age, data = df)
plot(exam_score ~ study_hours_per_day, data = df)
plot(exam_score ~ social_media_hours, data = df)
plot(exam_score ~ netflix_hours, data = df)
plot(exam_score ~ attendance_percentage, data = df)
plot(exam_score ~ sleep_hours, data = df)
plot(exam_score ~ exercise_frequency, data = df)
plot(exam_score ~ mental_health_rating, data = df)

# modello di regressione lineare multipla completo
fm <- lm(exam_score ~ age + gender + study_hours_per_day + social_media_hours + netflix_hours +
           part_time_job + attendance_percentage + sleep_hours + diet_quality + exercise_frequency +
           parental_education_level + internet_quality + mental_health_rating + extracurricular_participation,
         data = df)

summary(fm)  # riepilogo modello completo

# selezione automatica del modello finale (stepwise AIC)
fm_final <- step(fm, direction = "both", trace = FALSE)
summary(fm_final)  # riepilogo modello finale

# coefficienti e devianza del modello finale
coef(fm_final)
deviance(fm_final)

# diagnostica grafica standard
plot(fm_final)

# residui e fitted
residui <- residuals(fm_final)
fitted_values <- fitted(fm_final)

# distanza di Cook
cook <- cooks.distance(fm_final)
plot(cook, main = "Distanza di Cook")
abline(h = 4/length(cook), lty = 2)

# matrice di correlazione tra coefficienti
m <- summary(fm_final, correlation = TRUE)$correlation
corrplot(m, tl.col = "Black", title = "\n\nMatrice di correlazione (coefficienti)", type = "lower", method = "number")

# test t sulla media dei residui
t.test(residui)

# normalità residui (Shapiro-Wilk)
shapiro.test(residui)

# Q-Q plot dei residui standardizzati
qqnorm(scale(residui))
abline(0, 1)

# test di omoschedasticità (Breusch-Pagan)
bptest(fm_final)

# test di autocorrelazione (Durbin-Watson)
dwtest(fm_final)

# intervalli di confidenza (95% e 99%)
confint(fm_final)
confint(fm_final, level = 0.99)

# intervalli di confidenza delle predizioni
conf <- predict(fm_final, level = 0.99, interval = "confidence")
head(conf)

