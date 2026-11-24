
pub struct MultipleLinearRegression {
    weights: Vec<f32>,
    bias: f32,
    lr: f32,
    epoch: usize,
}

impl MultipleLinearRegression {
    pub fn new(lr: f32, epoch: usize) -> Self {
        let w: Vec<f32> = Vec::new();
        MultipleLinearRegression {
            weights: w,
            bias: 0.0,
            epoch: epoch,
            lr: lr,
        }
    }

    pub fn predict(&self, x: &[Vec<f32>]) -> Vec<f32> {
        let mut pred: Vec<f32> = Vec::new();

        for item in x {
            assert_eq!(item.len(), self.weights.len(), "predict : chaque ligne doit avoir autant de colonnes que de poids");
            let y = item
                .iter()
                .zip(&self.weights)
                .map(|(i, w)| i * w)
                .sum::<f32>()
                + self.bias;
            pred.push(y);
        }

        pred
    }

    pub fn fit(&mut self, x: &[Vec<f32>], y: &[f32]) {
        assert!(!x.is_empty(), "Erreur : X ne doit pas être vide");
        assert_eq!(x.len() , y.len(), "Erreur : pas la même longueur");


        let n_samples = x.len();
        let n_features = x[0].len();

        self.weights = vec![0.0; n_features];
        self.bias = 0.0;

        for _ in 0..self.epoch {
            let predict = self.predict(x);

            let errors: Vec<f32> = predict.iter().zip(y).map(|(p, yi)| p - yi).collect();

            for j in 0..n_features {
                let mut sum = 0.0;
                for i in 0..n_samples {
                    sum += x[i][j] * errors[i];
                }
                let gradient = 2.0 * sum / n_samples as f32;
                self.weights[j] -= self.lr * gradient;
            }

            let mean_error:f32 = errors.iter().sum::<f32>() / errors.len() as f32;
            let gradient_b = 2.0 * mean_error;

            self.bias -= self.lr * gradient_b;

            
        }
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn bias(&self) -> f32 {
        self.bias
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

        let pred = model.predict(&[vec![1.0, 1.0], vec![2.0, 0.0]]);

        assert!((pred[0] - 6.0).abs() < 1e-5, "attendu 6, obtenu {}", pred[0]); // 2×1 + 3×1 + 1
        assert!((pred[1] - 5.0).abs() < 1e-5, "attendu 5, obtenu {}", pred[1]); // 2×2 + 3×0 + 1
    }

    #[test]
    fn fit_retrouve_les_poids_deux_colonnes() {
        // y = 2·x1 + 3·x2 + 5
        let (x, y) = generate(500, &[2.0, 3.0], 5.0, 42);

        let mut model = MultipleLinearRegression::new(0.01, 5000);
        model.fit(&x, &y);

        let w = model.weights();
        assert!((w[0] - 2.0).abs() < EPS, "w1 attendu 2, obtenu {}", w[0]);
        assert!((w[1] - 3.0).abs() < EPS, "w2 attendu 3, obtenu {}", w[1]);
        assert!((model.bias() - 5.0).abs() < EPS, "biais attendu 5, obtenu {}", model.bias());
    }

    #[test]
    fn fit_une_colonne_comme_regression_simple() {
        // Avec une seule colonne, on doit retrouver y = 4x + 1,
        // comme le ferait get_coefficient.
        let (x, y) = generate(300, &[4.0], 1.0, 7);

        let mut model = MultipleLinearRegression::new(0.01, 5000);
        model.fit(&x, &y);

        assert!((model.weights()[0] - 4.0).abs() < EPS, "poids attendu 4, obtenu {}", model.weights()[0]);
        assert!((model.bias() - 1.0).abs() < EPS, "biais attendu 1, obtenu {}", model.bias());
    }

    #[test]
    fn fit_puis_predict_sur_nouvelles_donnees() {
        let (x_train, y_train) = generate(500, &[2.0, 3.0], 5.0, 42);
        let (x_test, y_test) = generate(50, &[2.0, 3.0], 5.0, 99);

        let mut model = MultipleLinearRegression::new(0.01, 5000);
        model.fit(&x_train, &y_train);
        let pred = model.predict(&x_test);

        for (p, yt) in pred.iter().zip(&y_test) {
            assert!((p - yt).abs() < 1e-2, "prédiction {p}, attendu {yt}");
        }
    }

    #[test]
    #[should_panic(expected = "ne doit pas être vide")]
    fn fit_x_vide() {
        let x: Vec<Vec<f32>> = vec![];
        let y: Vec<f32> = vec![];

        MultipleLinearRegression::new(0.01, 10).fit(&x, &y);
    }

    #[test]
    #[should_panic(expected = "pas la même longueur")]
    fn fit_longueurs_differentes() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let y = vec![1.0];

        MultipleLinearRegression::new(0.01, 10).fit(&x, &y);
    }

    #[test]
    #[should_panic(expected = "autant de colonnes que de poids")]
    fn predict_avant_fit() {
        let model = MultipleLinearRegression::new(0.01, 10);

        model.predict(&[vec![1.0, 2.0]]);
    }

    #[test]
    #[should_panic(expected = "autant de colonnes que de poids")]
    fn predict_mauvais_nombre_de_colonnes() {
        let (x, y) = generate(100, &[2.0, 3.0], 5.0, 1);
        let mut model = MultipleLinearRegression::new(0.01, 100);
        model.fit(&x, &y);

        // Entraîné sur 2 colonnes, on lui en donne 3
        model.predict(&[vec![1.0, 2.0, 3.0]]);
    }
}