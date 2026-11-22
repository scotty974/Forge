pub fn get_mean(arr: &[f32]) -> Option<f32> {
    if arr.is_empty() {
        return None;
    }

    let sum: f32 = arr.iter().sum();
    Some(sum / arr.len() as f32)
}

pub fn prediction_function(m: f32, x: f32, c: f32) -> f32 {
    let y: f32 = (m * x) + c;
    y
}

pub fn covariance(x: &[f32], x_mean: f32, y: &[f32], y_mean: f32, n: f32) -> f32 {
    let cov: f32 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
        .sum::<f32>()
        / n;

    cov
}

pub fn variance(arr: &[f32], mean: f32, n: f32) -> f32 {
    let var: f32 = arr.iter().map(|xi| (xi - mean).powi(2)).sum::<f32>() / n;

    var
}

pub fn get_coefficient(x: &[f32], y: &[f32]) -> (f32, f32) {
    let x_mean = get_mean(x).unwrap();
    let y_mean = get_mean(y).unwrap();
    let n: f32 = x.len() as f32;

    let m = covariance(x, x_mean, y, y_mean, n) / variance(x, x_mean, n);
    let b = y_mean - x_mean * m;

    (m, b)
}

#[cfg(test)]
mod test {
    use crate::utils::calcul_linear::variance;

    use super::*;

    #[test]
    fn prediction_function_test() {
        assert_eq!(prediction_function(2.0, 3.0, 1.0), 7.0);
        assert_eq!(prediction_function(5.0, 2.0, 10.0), 20.0);
        assert_eq!(prediction_function(1.0, 10.0, 0.0), 10.0);
        assert_eq!(prediction_function(0.0, 100.0, 5.0), 5.0);
        assert_eq!(prediction_function(-2.0, 3.0, 4.0), -2.0);
    }

    #[test]
    fn covariance_test() {
        let x = vec![1.0, 3.0];
        let x_mean = 2.0;
        let y = vec![2.0, 6.0];
        let y_mean = 4.0;

        let n = 2.0;

        let result = covariance(&x, x_mean, &y, y_mean, n);

        assert_eq!(result, 2.0)
    }

    #[test]
    fn variance_test() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let n = 5.0;

        assert_eq!(variance(&x, mean, n), 2.0);
    }

    
}
