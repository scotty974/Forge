use crate::{error::error::ForgeError, utils::calcul_linear::{get_mean, variance}};

pub struct StandardScaler {
    mean: Vec<f32>,
    stds: Vec<f32>

}

impl StandardScaler {
    pub fn new() -> Self {
        StandardScaler { mean: Vec::new(), stds: Vec::new() }
    }

    pub fn fit(&mut self, data: &[Vec<f32>]) -> Result<(), ForgeError> {

        assert!(!data.is_empty(), "scaler : data is empty");
        let n_sample = data.len();
        let n_features = data[0].len();

        self.mean = Vec::with_capacity(n_features);
        self.stds = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let col: Vec<f32> = data.iter().map(|row| row[j]).collect();

            let mean_col = get_mean(&col)?;

            let s = variance(
                &col,
                mean_col,
                n_sample as f32,
            )
            .sqrt();

        assert_ne!(s, 0.0, "scaler : ecart type 0 ");

        self.mean.push(mean_col);
        self.stds.push(s);
        }
        Ok(())
    }

    pub fn transform(&self, data: &[Vec<f32>]) -> Vec<Vec<f32>>{
        let mut result: Vec<Vec<f32>> = Vec::with_capacity(data.len());
        for row in data{
            assert_eq!(row.len(), self.mean.len(), "transform : data len is different of row len");
            let mut new_row: Vec<f32> = Vec::with_capacity(row.len());
            for j in 0..row.len(){
                let z = (row[j] - self.mean[j]) / self.stds[j];
                new_row.push(z);
            }
            result.push(new_row)
        }

        result
    }

    pub fn mean_(&self) -> Vec<f32> {
        self.mean.clone()
    }
}
#[cfg(test)]
mod test {
    use super::*;

    const EPS: f32 = 1e-4;

    /// Extrait la colonne j d'une matrice rangée par lignes.
    fn colonne(data: &[Vec<f32>], j: usize) -> Vec<f32> {
        data.iter().map(|row| row[j]).collect()
    }

    /// Moyenne et écart-type (population, divisé par n) d'une colonne.
    fn moyenne_ecart_type(col: &[f32]) -> (f32, f32) {
        let n = col.len() as f32;
        let m = col.iter().sum::<f32>() / n;
        let s = (col.iter().map(|v| (v - m).powi(2)).sum::<f32>() / n).sqrt();
        (m, s)
    }

    // ---------- fit ----------

    #[test]
    fn fit_calcule_moyenne_et_ecart_type_a_la_main() {
        // Colonne [1, 2, 3] : moyenne 2, écart-type racine(2/3) ≈ 0,8165
        let data = vec![vec![1.0], vec![2.0], vec![3.0]];

        let mut scaler = StandardScaler::new();
        scaler.fit(&data);

        assert!((scaler.mean[0] - 2.0).abs() < EPS);
        assert!((scaler.stds[0] - (2.0f32 / 3.0).sqrt()).abs() < EPS);
    }

    #[test]
    fn fit_une_moyenne_par_colonne() {
        // Colonne 0 : [1, 2, 3] → moyenne 2 ; colonne 1 : [10, 20, 30] → moyenne 20
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];

        let mut scaler = StandardScaler::new();
        scaler.fit(&data);

        assert_eq!(scaler.mean_(), vec![2.0, 20.0]);
        assert_eq!(scaler.stds.len(), 2);
    }

    #[test]
    fn fit_deux_fois_remplace_les_valeurs() {
        let mut scaler = StandardScaler::new();
        scaler.fit(&[vec![1.0], vec![3.0]]);
        scaler.fit(&[vec![10.0], vec![30.0]]);

        // Pas d'accumulation : une seule moyenne, celle du second fit
        assert_eq!(scaler.mean_(), vec![20.0]);
    }

    // ---------- transform ----------

    #[test]
    fn transform_valeurs_calculees_a_la_main() {
        // [1, 2, 3] → (x - 2) / 0,8165 ≈ [-1,2247 ; 0 ; 1,2247]
        let data = vec![vec![1.0], vec![2.0], vec![3.0]];

        let mut scaler = StandardScaler::new();
        scaler.fit(&data);
        let z = scaler.transform(&data);

        assert!((z[0][0] + 1.2247).abs() < EPS, "obtenu {}", z[0][0]);
        assert!(z[1][0].abs() < EPS, "obtenu {}", z[1][0]);
        assert!((z[2][0] - 1.2247).abs() < EPS, "obtenu {}", z[2][0]);
    }

    #[test]
    fn transform_donne_moyenne_0_et_ecart_type_1() {
        // Trois colonnes d'échelles très différentes, comme TV / radio / presse
        let data: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                let i = i as f32;
                vec![i * 3.0, 50.0 - i * 0.4, (i * 7.0) % 13.0 + 100.0]
            })
            .collect();

        let mut scaler = StandardScaler::new();
        scaler.fit(&data);
        let z = scaler.transform(&data);

        for j in 0..3 {
            let (m, s) = moyenne_ecart_type(&colonne(&z, j));
            assert!(m.abs() < EPS, "colonne {j} : moyenne {m} au lieu de 0");
            assert!((s - 1.0).abs() < EPS, "colonne {j} : écart-type {s} au lieu de 1");
        }
    }

    #[test]
    fn transform_test_utilise_les_valeurs_du_train() {
        // Train : [0, 2, 4] → moyenne 2, écart-type racine(8/3) ≈ 1,633
        let train = vec![vec![0.0], vec![2.0], vec![4.0]];
        let test = vec![vec![2.0], vec![6.0]];

        let mut scaler = StandardScaler::new();
        scaler.fit(&train);
        let z = scaler.transform(&test);

        let s = (8.0f32 / 3.0).sqrt();
        // 2 est la moyenne du train → 0
        assert!(z[0][0].abs() < EPS, "obtenu {}", z[0][0]);
        // 6 → (6 - 2) / 1,633 ≈ 2,449, calculé avec les valeurs du train, pas du test
        assert!((z[1][0] - 4.0 / s).abs() < EPS, "obtenu {}", z[1][0]);
    }

    #[test]
    fn transform_ne_modifie_pas_les_donnees_d_origine() {
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        let copie = data.clone();

        let mut scaler = StandardScaler::new();
        scaler.fit(&data);
        let _ = scaler.transform(&data);

        assert_eq!(data, copie);
    }

    #[test]
    fn transform_garde_la_forme_de_la_matrice() {
        let data = vec![vec![1.0, 10.0, 5.0], vec![2.0, 20.0, 7.0]];

        let mut scaler = StandardScaler::new();
        scaler.fit(&data);
        let z = scaler.transform(&data);

        assert_eq!(z.len(), 2);
        assert!(z.iter().all(|row| row.len() == 3));
    }

    // ---------- Cas qui doivent paniquer ----------

    #[test]
    #[should_panic(expected = "data is empty")]
    fn fit_donnees_vides() {
        let data: Vec<Vec<f32>> = vec![];
        StandardScaler::new().fit(&data);
    }

    #[test]
    #[should_panic(expected = "ecart type 0")]
    fn fit_colonne_constante() {
        // La colonne 1 vaut toujours 5 : écart-type nul
        let data = vec![vec![1.0, 5.0], vec![2.0, 5.0], vec![3.0, 5.0]];
        StandardScaler::new().fit(&data);
    }

    #[test]
    #[should_panic(expected = "data len is different of row len")]
    fn transform_avant_fit() {
        let scaler = StandardScaler::new();
        scaler.transform(&[vec![1.0, 2.0]]);
    }

    #[test]
    #[should_panic(expected = "data len is different of row len")]
    fn transform_mauvais_nombre_de_colonnes() {
        let mut scaler = StandardScaler::new();
        scaler.fit(&[vec![1.0, 2.0], vec![3.0, 4.0]]);

        // Appris sur 2 colonnes, on lui en donne 3
        scaler.transform(&[vec![1.0, 2.0, 3.0]]);
    }
}