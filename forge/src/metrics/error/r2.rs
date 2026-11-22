pub fn r2(y: &[f32], y_pred: &[f32]) -> f32 {

    assert_eq!(y.len(), y_pred.len(), "Pas de même longueur");
    assert!(!y.is_empty(), "y ne dois pas être vide");

    let sum: f32 = y.iter().sum();
    let y_mean: f32 = sum / y.len() as f32;
    let r2 = 1.0
        - y.iter()
            .zip(y_pred)
            .map(|(ya, yi)| (ya - yi).powi(2))
            .sum::<f32>()
            / y.iter().map(|ya| (ya - y_mean).powi(2)).sum::<f32>();

    r2
}
