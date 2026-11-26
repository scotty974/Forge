use crate::utils::calcul_linear::{get_mean, variance};

pub struct StandardScaler {
    mean: Vec<f32>,
    stds: Vec<f32>,
}

impl StandardScaler {
    pub fn new() -> Self {
        StandardScaler { mean: Vec::new(), stds: Vec::new() }
    }

    pub fn fit(&mut self, data: &[Vec<f32>]) {

        assert!(!data.is_empty(), "scaler : data is empty");
        let n_sample = data.len();
        let n_features = data[0].len();

        self.mean = Vec::with_capacity(n_features);
        self.stds = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let col: Vec<f32> = data.iter().map(|row| row[j]).collect();

            let mean_col = get_mean(&col).expect("scaler : get mean by col ");

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
