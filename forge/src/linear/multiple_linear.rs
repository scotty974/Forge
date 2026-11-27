use crate::{error::error::ForgeError, metrics::error::mse::mean_squared_error};

pub struct MultipleLinearRegression {
    weights: Vec<f32>,
    bias: f32,
    lr: f32,
    epoch: usize,
    loss_history: Vec<f32>,
    tol: Option<f32>,
}

impl MultipleLinearRegression {
    pub fn new(lr: f32, epoch: usize) -> Self {
        let w: Vec<f32> = Vec::new();
        MultipleLinearRegression {
            weights: w,
            bias: 0.0,
            epoch: epoch,
            lr: lr,
            loss_history: Vec::new(),
            tol: None,
        }
    }

    pub fn predict(&self, x: &[Vec<f32>]) -> Result<Vec<f32>, ForgeError> {
        let mut pred: Vec<f32> = Vec::new();

        for item in x {
            if item.len() != self.weights.len() {
                return Err(ForgeError::LengthMismatch {
                    context: "MultipleLinearRegression::predict",
                    left: item.len(),
                    right: self.weights.len(),
                });
            }
            let y = item
                .iter()
                .zip(&self.weights)
                .map(|(i, w)| i * w)
                .sum::<f32>()
                + self.bias;
            pred.push(y);
        }

        Ok(pred)
    }

    pub fn fit(&mut self, x: &[Vec<f32>], y: &[f32]) -> Result<(), ForgeError> {
        if x.is_empty() {
            return Err(ForgeError::EmptyData {
                context: "MultipleLinearRegression::fit",
            });
        }

        if x.len() != y.len() {
            return Err(ForgeError::LengthMismatch {
                context: "MultipleLinearRegression::fit",
                left: x.len(),
                right: y.len(),
            });
        }

        self.loss_history.clear();
        let n_samples = x.len();
        let n_features = x[0].len();

        self.weights = vec![0.0; n_features];
        self.bias = 0.0;

        for epoch in 0..self.epoch {
            let predict = self.predict(x)?;

            let errors: Vec<f32> = predict.iter().zip(y).map(|(p, yi)| p - yi).collect();

            for j in 0..n_features {
                let mut sum = 0.0;
                for i in 0..n_samples {
                    sum += x[i][j] * errors[i];
                }
                let gradient = 2.0 * sum / n_samples as f32;
                self.weights[j] -= self.lr * gradient;
            }
            let mse = mean_squared_error(y, &predict)?;
            if !mse.is_finite() {
                return Err(ForgeError::Diverged { epoch });
            }
            let mean_error: f32 = errors.iter().sum::<f32>() / errors.len() as f32;
            let gradient_b = 2.0 * mean_error;

            self.bias -= self.lr * gradient_b;
            self.loss_history.push(mse);
            if let Some(tol) = self.tol {
                let n = self.loss_history.len();
                if n >= 2 {
                    let amelioration = self.loss_history[n - 2] - self.loss_history[n - 1];
                    if amelioration < tol {
                        break;
                    }
                }
            }
        }
        Ok(())
    }
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tol = Some(tol);
        self
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn bias(&self) -> f32 {
        self.bias
    }

    pub fn loss_history(&self) -> &[f32] {
        &self.loss_history
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const EPS: f32 = 1e-3;

    /// Génère n lignes de `weights.len()` colonnes dans [0, 10],
    /// avec y = somme(w_j * x_j) + bias, sans bruit. Même seed = mêmes données.
    fn generate(n: usize, weights: &[f32], bias: f32, seed: u64) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / ((1u64 << 31) as f32)
        };

        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let row: Vec<f32> = weights.iter().map(|_| next() * 10.0).collect();
            let target = row.iter().zip(weights).map(|(xi, wi)| xi * wi).sum::<f32>() + bias;
            x.push(row);
            y.push(target);
        }
        (x, y)
    }

    #[test]
    fn predict_avec_poids_fixes() {
        let mut model = MultipleLinearRegression::new(0.01, 100);
        model.weights = vec![2.0, 3.0];
        model.bias = 1.0;

        let pred = model.predict(&[vec![1.0, 1.0], vec![2.0, 0.0]]).unwrap();

        assert!(
            (pred[0] - 6.0).abs() < 1e-5,
            "attendu 6, obtenu {}",
            pred[0]
        ); // 2×1 + 3×1 + 1
        assert!(
            (pred[1] - 5.0).abs() < 1e-5,
            "attendu 5, obtenu {}",
            pred[1]
        ); // 2×2 + 3×0 + 1
    }

    #[test]
    fn fit_retrouve_les_poids_deux_colonnes() {
        // y = 2·x1 + 3·x2 + 5
        let (x, y) = generate(500, &[2.0, 3.0], 5.0, 42);

        let mut model = MultipleLinearRegression::new(0.01, 5000);
        model.fit(&x, &y).unwrap();

        let w = model.weights();
        assert!((w[0] - 2.0).abs() < EPS, "w1 attendu 2, obtenu {}", w[0]);
        assert!((w[1] - 3.0).abs() < EPS, "w2 attendu 3, obtenu {}", w[1]);
        assert!(
            (model.bias() - 5.0).abs() < EPS,
            "biais attendu 5, obtenu {}",
            model.bias()
        );
    }

    #[test]
    fn fit_une_colonne_comme_regression_simple() {
        // Avec une seule colonne, on doit retrouver y = 4x + 1,
        // comme le ferait get_coefficient.
        let (x, y) = generate(300, &[4.0], 1.0, 7);

        let mut model = MultipleLinearRegression::new(0.01, 5000);
        model.fit(&x, &y).unwrap();

        assert!(
            (model.weights()[0] - 4.0).abs() < EPS,
            "poids attendu 4, obtenu {}",
            model.weights()[0]
        );
        assert!(
            (model.bias() - 1.0).abs() < EPS,
            "biais attendu 1, obtenu {}",
            model.bias()
        );
    }

    #[test]
    fn fit_puis_predict_sur_nouvelles_donnees() {
        let (x_train, y_train) = generate(500, &[2.0, 3.0], 5.0, 42);
        let (x_test, y_test) = generate(50, &[2.0, 3.0], 5.0, 99);

        let mut model = MultipleLinearRegression::new(0.01, 5000);
        model.fit(&x_train, &y_train).unwrap();
        let pred = model.predict(&x_test).unwrap();

        for (p, yt) in pred.iter().zip(&y_test) {
            assert!((p - yt).abs() < 1e-2, "prédiction {p}, attendu {yt}");
        }
    }

    #[test]
    fn fit_x_vide() {
        let x: Vec<Vec<f32>> = vec![];
        let y: Vec<f32> = vec![];

        let result = MultipleLinearRegression::new(0.01, 10).fit(&x, &y);

        assert!(matches!(result, Err(ForgeError::EmptyData { .. })));
    }

    #[test]
    fn fit_longueurs_differentes() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let y = vec![1.0];

        let result = MultipleLinearRegression::new(0.01, 10).fit(&x, &y);

        assert!(matches!(result, Err(ForgeError::LengthMismatch { .. })));
    }

    #[test]
    fn predict_avant_fit() {
        let model = MultipleLinearRegression::new(0.01, 10);

        let result = model.predict(&[vec![1.0, 2.0]]);

        assert!(matches!(result, Err(ForgeError::LengthMismatch { .. })));
    }

    #[test]
    fn predict_mauvais_nombre_de_colonnes() {
        let (x, y) = generate(100, &[2.0, 3.0], 5.0, 1);
        let mut model = MultipleLinearRegression::new(0.01, 100);
        model.fit(&x, &y).unwrap();

        // Entraîné sur 2 colonnes, on lui en donne 3
        let result = model.predict(&[vec![1.0, 2.0, 3.0]]);

        assert!(matches!(result, Err(ForgeError::LengthMismatch { .. })));
    }

    #[test]
    fn early_stopping_arrete_avant_la_fin() {
        let (x, y) = generate(200, &[2.0, 3.0], 5.0, 42);

        let mut model = MultipleLinearRegression::new(0.01, 100_000).with_tolerance(1e-7);
        model.fit(&x, &y).unwrap();

        assert!(
            model.loss_history().len() < 100_000,
            "le modèle aurait dû s'arrêter avant"
        );
        assert!((model.weights()[0] - 2.0).abs() < 1e-2);
    }
}