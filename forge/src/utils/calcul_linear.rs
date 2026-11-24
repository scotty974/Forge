pub fn get_mean(arr: &[f32]) -> Option<f32> {
    if arr.is_empty() {
        return None;
    }

    let sum: f32 = arr.iter().sum();
    Some(sum / arr.len() as f32)
}

pub fn prediction_function(m: f32, x: f32, c: f32) -> f32 {
    (m * x) + c
}

pub fn covariance(x: &[f32], x_mean: f32, y: &[f32], y_mean: f32, n: f32) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
        .sum::<f32>()
        / n
}

pub fn variance(arr: &[f32], mean: f32, n: f32) -> f32 {
    arr.iter().map(|xi| (xi - mean).powi(2)).sum::<f32>() / n
}

pub fn get_coefficient(x: &[f32], y: &[f32]) -> (f32, f32) {
    assert!(!x.is_empty(), "get_coefficient : x ne doit pas être vide");
    assert_eq!(
        x.len(),
        y.len(),
        "get_coefficient : x et y doivent avoir la même longueur"
    );

    // Les deux assert ci-dessus garantissent que x et y ne sont pas vides,
    // donc les unwrap ne peuvent pas paniquer.
    let x_mean = get_mean(x).unwrap();
    let y_mean = get_mean(y).unwrap();
    let n: f32 = x.len() as f32;

    let var = variance(x, x_mean, n);
    assert!(
        var != 0.0,
        "get_coefficient : variance de x nulle, tous les x sont identiques"
    );

    let cov = covariance(x, x_mean, y, y_mean, n);

    let m = cov / var;
    let b = y_mean - x_mean * m;

    (m, b)
}

#[cfg(test)]
mod test {
    use super::*;

    const EPS: f32 = 1e-5;

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
        let y = vec![2.0, 6.0];

        let result = covariance(&x, 2.0, &y, 4.0, 2.0);

        assert!((result - 2.0).abs() < EPS);
    }

    #[test]
    fn variance_test() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((variance(&x, 3.0, 5.0) - 2.0).abs() < EPS);
    }

    #[test]
    fn get_coefficient_droite_parfaite() {
        // Points alignés sur y = 2x + 1
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f32> = x.iter().map(|xi| 2.0 * xi + 1.0).collect();

        let (m, b) = get_coefficient(&x, &y);

        assert!((m - 2.0).abs() < EPS, "pente attendue 2, obtenue {m}");
        assert!((b - 1.0).abs() < EPS, "ordonnée attendue 1, obtenue {b}");
    }

    #[test]
    #[should_panic(expected = "variance de x nulle")]
    fn get_coefficient_x_identiques() {
        let x = vec![3.0, 3.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];

        get_coefficient(&x, &y);
    }

    #[test]
    #[should_panic(expected = "même longueur")]
    fn get_coefficient_longueurs_differentes() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];

        get_coefficient(&x, &y);
    }

    #[test]
    #[should_panic(expected = "x ne doit pas être vide")]
    fn get_coefficient_vide() {
        let x: Vec<f32> = vec![];
        let y: Vec<f32> = vec![];

        get_coefficient(&x, &y);
    }
}