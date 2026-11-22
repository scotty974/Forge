pub fn mean_squared_error(y: &[f32], y_pred: &[f32]) -> f32 {
    let n = y.len() as f32;


    let mse = y
    .iter()
    .zip(y_pred)
    .map(|(ya, yi)|{
        (ya - yi).abs().powi(2)
    })
    .sum::<f32>() / n;

    mse
}
